using LinearAlgebra

function losECI(r1, r2, Re::Real=6_378_137.0; margin::Real=0.0, tol::Real=1e-9)

    """
        losECI(r1, r2, Re=6_378_137.0; margin=0.0, tol=1e-9)

        Determina si dos objetos en ECI tienen línea de visión sin obstrucción por la Tierra (modelo esférico).

        # Argumentos
        - `r1`, `r2` : Vectores posición ECI (m), de longitud 3.
        - `Re`       : Radio terrestre (m). Por defecto 6_378_137.0 (WGS-84 aprox).
        - `margin`   : Margen extra (m) al radio para modelar atmósfera/seguridad. (def: 0.0)
        - `tol`      : Tolerancia numérica. (def: 1e-9)

        # Retorna
        - `visible::Bool` : `true` si hay LOS; `false` si la Tierra bloquea.
        - `details::Dict` : Información de diagnóstico (radio usado, normas, motivo, intervalos s).

        Tangencia dentro del segmento se considera bloqueo (conservador).
    """
    
    # ---- Validaciones básicas ----
    length(r1) == 3 || throw(ArgumentError("r1 debe tener longitud 3"))
    length(r2) == 3 || throw(ArgumentError("r2 debe tener longitud 3"))

    # Asegurar Float64 y forma "fila" no es necesario en Julia; trabajamos con vectores columna 3
    r1v = Vector{Float64}(r1)
    r2v = Vector{Float64}(r2)

    Rb  = float(Re) + float(margin)
    Rb2 = Rb*Rb

    n1 = dot(r1v, r1v)
    n2 = dot(r2v, r2v)

    details = Dict{Symbol,Any}(
        :RadiusUsed  => Rb,
        :r1_norm     => sqrt(n1),
        :r2_norm     => sqrt(n2),
        :BlockReason => ""
    )

    # Algún punto por debajo de la superficie
    if details[:r1_norm] < Re - tol || details[:r2_norm] < Re - tol
        details[:BlockReason] = "Endpoint below surface"
        return false, details
    end

    # Segmento y coeficientes de la cuadrática
    d = r2v .- r1v
    a = dot(d,d)
    if a <= tol
        details[:BlockReason] = "Coincident points"
        return false, details
    end
    b = 2.0*dot(r1v, d)
    c = n1 - Rb2
    disc = b*b - 4.0*a*c

    # Línea infinita NO intersecta esfera (con tolerancia robusta)
    if disc < -abs(b*b + 4.0*a*abs(c))*tol
        details[:BlockReason] = "No line-sphere intersection"
        return true, details
    end

    # Tangencia (disc ≈ 0)
    if abs(disc) <= max(1.0, b*b + 4.0*a*abs(c))*tol
        s = -b/(2.0*a)
        details[:s_intervals] = s
        if s >= -tol && s <= 1.0 + tol
            details[:BlockReason] = "Grazing tangency at Earth limb"
            return false, details
        else
            details[:BlockReason] = "Tangency outside segment"
            return true, details
        end
    end

    # Dos intersecciones: comprobar solape con [0,1]
    sqrtDisc = sqrt(max(disc, 0.0))
    s1 = (-b - sqrtDisc)/(2.0*a)
    s2 = (-b + sqrtDisc)/(2.0*a)
    smin = min(s1, s2)
    smax = max(s1, s2)

    intersectsSegment = !(smax < 0.0 - tol || smin > 1.0 + tol)

    details[:s_intervals] = (smin, smax)
    if intersectsSegment
        details[:BlockReason] = "Earth occlusion (segment intersects sphere)"
        return false, details
    else
        details[:BlockReason] = "Intersections lie outside segment"
        return true, details
    end
end
