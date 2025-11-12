using LinearAlgebra

function eclipseEarthECI(r_sc, r_sun; Re=6_378_137.0, Rs=696_340_000.0,
                        Margin=0.0, Tol=1e-9, Model="conical")

    """
        eclipseEarthECI(r_sc, r_sun; Re=6_378_137.0, Rs=696_340_000.0, Margin=0.0, Tol=1e-9, Model="conical")

        Determina si un satélite está en eclipse por la Tierra (umbra/penumbra) usando el modelo cónico.
        - `r_sc`, `r_sun`: vectores ECI (m) respecto al centro de la Tierra (longitud 3).
        - `Re`: radio terrestre (m). Por defecto 6_378_137.0.
        - `Rs`: radio solar (m). Por defecto 696_340_000.0.
        - `Margin`: margen extra (m) para “engordar” la sombra.
        - `Tol`: tolerancia numérica.
        - `Model`: `"conical"` (por defecto) o `"cylindrical"`.

        Retorna: `(inEclipse::Bool, state::String, details::Dict{Symbol,Any})`
    """

    # --- Validaciones básicas ---
    length(r_sc) == 3 || throw(ArgumentError("r_sc debe tener longitud 3"))
    length(r_sun) == 3 || throw(ArgumentError("r_sun debe tener longitud 3"))
    Re > 0 || throw(ArgumentError("Re debe ser positivo"))
    Rs > 0 || throw(ArgumentError("Rs debe ser positivo"))

    # Asegurar Float64
    r_sc = collect(Float64, r_sc)
    r_sun = collect(Float64, r_sun)

    # Magnitudes básicas
    Dsun = norm(r_sun)
    Dsun > 0 || throw(ArgumentError("r_sun inválido (norma cero)."))

    # Eje de sombra (desde la Tierra mirando “hacia la noche”)
    u = -r_sun / Dsun

    # Coordenadas del satélite respecto al eje
    x   = dot(r_sc, u)                 # proyección axial (m)
    rho = norm(r_sc .- x .* u)         # distancia radial al eje (m)

    details = Dict{Symbol,Any}(
        :u_axis   => u,
        :x_axis   => x,
        :rho_axis => rho,
        :Re_used  => Re + Margin,
        :Rs_used  => Rs,
        :Dsun     => Dsun,
        :Model    => lowercase(String(Model)),
    )

    # Si está del lado del Sol (x <= 0) -> iluminado
    if x <= 0
        details[:BlockReason] = "Sunward hemisphere (x<=0)"
        details[:r_umbra] = 0.0
        details[:r_penum] = 0.0
        return false, "lit", details
    end

    model = lowercase(String(Model))
    if model == "cylindrical"
        # Sombra cilíndrica de radio Re+Margin
        Rb = Re + Margin
        details[:r_umbra] = Rb
        details[:r_penum] = NaN
        if rho <= Rb + Tol
            details[:BlockReason] = "Cylindrical shadow (rho <= Re+Margin)"
            return true, "umbra", details   # sin distinguir penumbra en este modelo
        else
            details[:BlockReason] = "Outside cylindrical shadow"
            return false, "lit", details
        end
    else
        # ----- Modelo cónico -----
        # Longitudes características
        denom_umbra = (Rs - Re)
        denom_penum = (Rs + Re)
        (denom_umbra > 0 && denom_penum > 0) || throw(ArgumentError("Parámetros Rs/Re no válidos para el cono."))

        L  = Dsun * Re / denom_umbra      # longitud umbra
        Lp = Dsun * Re / denom_penum      # “longitud” penumbra

        # Radios en la sección a distancia x
        r_umbra = Re * (1 - x / L) + Margin
        r_umbra = max(r_umbra, 0.0)       # si x > L, umbra desaparece
        r_penum = Re * (1 + x / Lp) + Margin

        details[:L_umbra] = L
        details[:L_penum] = Lp
        details[:r_umbra] = r_umbra
        details[:r_penum] = r_penum

        if (x <= L + Tol) && (r_umbra > 0.0) && (rho <= r_umbra + Tol)
            details[:BlockReason] = "Inside umbra cone"
            return true, "umbra", details
        elseif rho <= r_penum + Tol
            details[:BlockReason] = "Inside penumbra cone"
            return true, "penumbra", details
        else
            details[:BlockReason] = "Outside shadow cones"
            return false, "lit", details
        end
    end
end
