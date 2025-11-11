using LinearAlgebra
using Statistics

function phong_model_entirely(phi, d, C_spec, C_diff, alpha, u_n, u_sun, u_obs, A)

"""
    phong_model_entirely(phi, d, C_spec, C_diff, alpha, u_n, u_sun, u_obs, A) -> Float64

Calcula la magnitud aparente `m_app` con un modelo tipo Phong, sumando contribuciones
por faceta.

Parámetros
----------
- `phi`   :: Real o AbstractVector    # Irradiancia solar [W/m^2] (escalar o 1×N / N)
- `d`     :: Real o AbstractVector    # Distancia faceta-observador [m] (esc. o 1×N / N)
- `C_spec`:: Real                     # Coeficiente especular
- `C_diff`:: Real                     # Albedo difuso (usa C_diff/π si quieres Lambert físico)
- `alpha` :: Real                     # Exponente Phong
- `u_n`   :: AbstractMatrix{<:Real}   # Normales 3×N (o N×3 que se transpone)
- `u_sun` :: AbstractVecOrMat{<:Real} # 3×1 o 3×N (o fila 1×3 -> se transpone)
- `u_obs` :: AbstractVecOrMat{<:Real} # 3×1 o 3×N (o fila 1×3 -> se transpone)
- `A`     :: Real o AbstractVector    # Áreas por faceta [m^2] (esc. o 1×N / N)

Devuelve
--------
- `m_app` :: Float64  # magnitud aparente (referencia de -26.7 y normalización por mean(phi))
"""

    epsc = 1e-12
    TOL  = 1e-9

    # --- SANEADO DE FORMAS ---
    # u_n: 3×N (si viene N×3, transpón)
    if size(u_n,1) != 3 && size(u_n,2) == 3
        u_n = permutedims(u_n)             # Nx3 -> 3xN
    end
    size(u_n,1) == 3 || error("u_n debe ser 3xN")

    # u_sun, u_obs: 3×1 o 3×N (si vienen 1×3, transpón)
    if ndims(u_sun) == 2 && size(u_sun,1) == 1 && size(u_sun,2) == 3
        u_sun = permutedims(u_sun)
    end
    if ndims(u_obs) == 2 && size(u_obs,1) == 1 && size(u_obs,2) == 3
        u_obs = permutedims(u_obs)
    end
    size(u_sun,1) == 3 || error("u_sun debe tener 3 componentes en la 1ª dimensión")
    size(u_obs,1) == 3 || error("u_obs debe tener 3 componentes en la 1ª dimensión")

    N = size(u_n, 2)

    # A: 1×N
    A_vec = A
    if isa(A, Number)
        A_vec = fill(float(A), N)
    else
        A_vec = vec(A)
        length(A_vec) == N || error("A debe tener longitud N = $N")
    end

    # d, phi como 1×N si son escalares
    d_vec = d
    if isa(d, Number)
        d_vec = fill(float(d), N)
    else
        d_vec = vec(d)
        length(d_vec) == N || error("d debe tener longitud N = $N")
    end

    phi_vec = phi
    if isa(phi, Number)
        phi_vec = fill(float(phi), N)
    else
        phi_vec = vec(phi)
        length(phi_vec) == N || error("phi debe tener longitud N = $N")
    end

    # --- NORMALIZACIONES ---
    # u_n por columnas
    nrm_n = sqrt.(sum(abs2, u_n; dims=1))
    u_n   = u_n ./ max.(nrm_n, epsc)

    # Replica u_sun / u_obs si son 3×1
    if ndims(u_sun) == 1 || size(u_sun,2) == 1
        u_sun = repeat(reshape(u_sun, 3, 1), 1, N)
    end
    if ndims(u_obs) == 1 || size(u_obs,2) == 1
        u_obs = repeat(reshape(u_obs, 3, 1), 1, N)
    end

    # Normalizar por columnas
    u_sun = u_sun ./ max.(sqrt.(sum(abs2, u_sun; dims=1)), epsc)
    u_obs = u_obs ./ max.(sqrt.(sum(abs2, u_obs; dims=1)), epsc)

    # --- ANGULARES ---
    cos_in = vec(sum(u_n .* u_sun; dims=1))   # incidencia
    cos_v  = vec(sum(u_n .* u_obs; dims=1))   # visibilidad

    ok_idx = findall(i -> cos_in[i] > TOL && cos_v[i] > TOL, eachindex(cos_in))
    if isempty(ok_idx)
        # misma convención que tu versión comentada: devolver 0 si no hay flujo
        return 0.0
    end

    # Subconjuntos
    u_n_ok    = u_n[:, ok_idx]
    u_sun_ok  = u_sun[:, ok_idx]
    u_obs_ok  = u_obs[:, ok_idx]
    cos_in_ok = cos_in[ok_idx]
    cos_v_ok  = cos_v[ok_idx]
    A_ok      = A_vec[ok_idx]
    d_ok      = d_vec[ok_idx]
    phi_ok    = phi_vec[ok_idx]

    # --- ESPECULAR PHONG ---
    # u_spec = 2 * (u_n .* cos_in) - u_sun  (por columnas)
    u_spec = 2 .* (u_n_ok .* reshape(cos_in_ok, 1, :)) .- u_sun_ok
    nrm_s  = sqrt.(sum(abs2, u_spec; dims=1))
    u_spec = u_spec ./ max.(nrm_s, epsc)

    cos_s  = max.(vec(sum(u_obs_ok .* u_spec; dims=1)), 0.0)

    # Nota: usa C_diff/pi si quieres Lambert físicamente correcto
    rho_spec  = C_spec .* (cos_s .^ alpha) ./ max.(cos_in_ok, epsc)
    F_sun_ok  = phi_ok .* (C_diff .+ rho_spec) .* cos_in_ok          # o (C_diff/π + rho_spec)
    F_obs_ok  = F_sun_ok .* A_ok .* cos_v_ok ./ (max.(d_ok, epsc) .^ 2)

    F_tot = sum(F_obs_ok)

    return -26.7 - 2.5*log10(F_tot / mean(phi_vec))
end
