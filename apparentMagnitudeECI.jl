using LinearAlgebra

include("eclipseEarthECI.jl")
include("sunGlareConeECI.jl")
include("losECI.jl")
include("phong_model_entirely.jl")

"""
    m_app = apparent_magnitude_ECI(r_obs, r_deb, r_sun, debris, DCM_b2i;
                                   phi=1361.0,
                                   Re=6_378_137.0,
                                   glareAngleDeg=5.0)

Calcula la magnitud de brillo aparente `m_app` que percibe el satélite observador
debido a un objeto de basura espacial, usando el modelo de Phong.

Argumentos
----------
- `r_obs` :: AbstractVector{<:Real}  # posición ECI del observador [m]
- `r_deb` :: AbstractVector{<:Real}  # posición ECI de la basura [m]
- `r_sun` :: AbstractVector{<:Real}  # posición ECI del Sol [m]
- `debris`:: Dict{Symbol,Any}        # diccionario con geometría y óptica de la basura
                                     # (contiene :normals, :areas, :alpha, :C_spec, :C_diff, ...)
- `DCM_b2i`:: AbstractMatrix{<:Real} # matriz actitud cuerpo→ECI (3×3)

Keywords
--------
- `phi`           : irradiancia solar [W/m²] en la basura (por defecto 1361 W/m²)
- `Re`            : radio terrestre [m] para `losECI`
- `glareAngleDeg` : semiancho del cono de deslumbramiento del Sol visto desde el observador [deg]

Devuelve
--------
- `m_app` :: Float64  # magnitud aparente; devuelve `Inf` si no hay señal (eclipse, sin LOS, deslumbramiento)
"""
function apparent_magnitude_ECI(r_obs::AbstractVector,
                                r_deb::AbstractVector,
                                r_sun::AbstractVector,
                                debris::Dict{Symbol,Any},
                                DCM_b2i::AbstractMatrix;
                                phi::Real = 1361.0,
                                Re::Real = 6_378_137.0,
                                glareAngleDeg::Real = 5.0)

    # --- Asegurarnos de que todo está en Float64 y con longitud 3 ---
    r_obs_v = Vector{Float64}(r_obs)
    r_deb_v = Vector{Float64}(r_deb)
    r_sun_v = Vector{Float64}(r_sun)

    @assert length(r_obs_v) == 3 "r_obs debe ser un vector de longitud 3"
    @assert length(r_deb_v) == 3 "r_deb debe ser un vector de longitud 3"
    @assert length(r_sun_v) == 3 "r_sun debe ser un vector de longitud 3"

    # --- 1) Comprobar si la basura está en eclipse (umbra/penumbra) ---
    in_eclipse, region, _details_ecl = eclipseEarthECI(r_deb_v, r_sun_v)
    if in_eclipse
        # Si está en umbra/penumbra retornamos magnitud infinita (sin señal útil)
        return Inf
    end

    # --- 2) Comprobar línea de visión observador–basura (ocultación por la Tierra) ---
    hasLOS, _details_los = losECI(r_obs_v, r_deb_v, Re)
    if !hasLOS
        return Inf
    end

    # --- 3) Comprobar deslumbramiento del observador por el Sol ---
    # Si el objeto está demasiado cerca de la dirección del Sol visto desde el observador,
    # asumimos que no se puede medir su brillo.
    blinded, _ang_deg, _details_glare = sunGlareConeECI(r_sun_v, r_obs_v, r_deb_v, glareAngleDeg)
    if blinded
        return Inf
    end

    # --- 4) Normales de las facetas en el sistema del cuerpo ---
    u_n_body_any = debris[:normals]

    # Aceptamos tanto 3×N como N×3 y lo convertimos a 3×N
    u_n_body =
        if size(u_n_body_any, 1) == 3
            Array{Float64}(u_n_body_any)
        elseif size(u_n_body_any, 2) == 3
            permutedims(u_n_body_any)  # (3, N)
        else
            error("debris[:normals] debe ser de tamaño 3×N o N×3")
        end

    # Asegurar que son unitarias (por si acaso)
    norms = sqrt.(sum(abs2, u_n_body; dims = 1))
    norms = max.(norms, eps(Float64))
    u_n_body ./= norms

    # --- 5) Rotar normales a ECI usando la actitud (cuerpo→ECI) ---
    D = Array{Float64}(DCM_b2i)
    @assert size(D) == (3,3) "DCM_b2i debe ser una matriz 3×3"

    u_n_eci = D * u_n_body  # 3×N

    # --- 6) Direcciones Sol y observador desde la basura ---
    v_sun = r_sun_v .- r_deb_v  # vector basura→Sol
    v_obs = r_obs_v .- r_deb_v  # vector basura→observador

    u_sun_vec = v_sun / norm(v_sun)
    u_obs_vec = v_obs / norm(v_obs)

    Nfacets = size(u_n_eci, 2)

    # Repetimos dirección de Sol y observador para cada faceta
    u_sun = repeat(reshape(u_sun_vec, 3, 1), 1, Nfacets)  # 3×N
    u_obs = repeat(reshape(u_obs_vec, 3, 1), 1, Nfacets)  # 3×N

    # Distancia basura–observador (la misma para todas las facetas)
    d = norm(v_obs)

    # Áreas de faceta
    A = debris[:areas]
    A_vec =
        if length(A) == Nfacets
            Vector{Float64}(A)
        else
            error("debris[:areas] debe tener longitud igual al número de facetas")
        end

    # Coeficientes ópticos globales para el modelo Phong
    alpha  = debris[:alpha]
    C_spec = debris[:C_spec]
    C_diff = debris[:C_diff]

    # --- 7) Evaluar modelo de Phong (ya devuelve magnitud aparente) ---
    m_app = phong_model_entirely(phi, d, C_spec, C_diff, alpha,
                                u_n_eci, u_sun, u_obs, A_vec)

    return m_app
end
