using LinearAlgebra

function sunGlareConeECI(r_sun, r_obs, r_so, angMaxDeg; TolDeg=1e-6)

    """
        sunGlareConeECI(r_sun, r_obs, r_so, angMaxDeg; TolDeg=1e-6)

        Determina si el observador está "cegado" por el Sol porque el objeto observado
        está dentro del cono de contraluz.

        Definiciones (todas en ECI):
        - vSun = Observador→Sol = r_sun - r_obs
        - vObj = Observador→Objeto = r_so - r_obs

        Criterio:
        - ang = angle(vSun, vObj) en grados
        - blinded = (ang ≤ angMaxDeg + TolDeg)

        Parámetros:
        - `r_sun, r_obs, r_so` : vectores de longitud 3 (m)
        - `angMaxDeg`          : umbral angular en grados
        - `TolDeg` (kwarg)     : tolerancia en grados (def: 1e-6)

        Retorna:
        - `blinded::Bool`, `ang_deg::Float64`, `details::Dict{Symbol,Any}`
    """

    # Validaciones básicas
    (length(r_sun) == 3 && length(r_obs) == 3 && length(r_so) == 3) ||
        throw(ArgumentError("r_sun, r_obs y r_so deben tener longitud 3"))
    isa(angMaxDeg, Real) || throw(ArgumentError("angMaxDeg debe ser real"))
    TolDeg ≥ 0 || throw(ArgumentError("TolDeg debe ser ≥ 0"))

    # Asegurar Float64
    r_sun = collect(Float64, r_sun)
    r_obs = collect(Float64, r_obs)
    r_so  = collect(Float64, r_so)

    # Direcciones desde el observador
    vSun = r_sun .- r_obs       # Obs->Sol
    vObj = r_so  .- r_obs       # Obs->Objeto

    nSun = norm(vSun)
    nObj = norm(vObj)
    (nSun > 0 && nObj > 0) || throw(ArgumentError("Vectores degenerados: r_sun y r_so no deben coincidir con r_obs."))

    uSun = vSun ./ nSun
    uObj = vObj ./ nObj

    # Ángulo entre direcciones
    c = dot(uSun, uObj)
    c = clamp(c, -1.0, 1.0)     # robustez numérica
    ang_deg = acosd(c)

    # Criterio de cegado
    blinded = ang_deg <= (float(angMaxDeg) + TolDeg)

    # Detalles
    details = Dict{Symbol,Any}(
        :u_obs_to_sun => uSun,
        :u_obs_to_obj => uObj,
        :dot_uSun_uObj => c,
        :ang_deg => ang_deg,
        :angMaxDeg => float(angMaxDeg),
        :TolDeg => TolDeg,
    )

    return blinded, ang_deg, details
end
