module DebrisObject

using LinearAlgebra
using Random
using Statistics

include("generateRandomShape.jl") 

"""
    Debris = Dict{Symbol,Any}

Diccionario que representa un objeto de basura espacial.

Campos principales típicos:

- :shape_type      :: Int              # 1..7 (prisma, esfera, placa, cilindro, cono, tronco, fragmento)
- :sat_size        :: Int              # categoría de tamaño (1: cubesat / pequeño, 2: <100kg, 3: grande)
- :sub_cat         :: Int              # subcategoría (1..4)
- :deploy_panels   :: Bool
- :alpha           :: Float64          # ancho del lóbulo especular (Phong)
- :C_abs           :: Float64          # coeficiente de absorción efectivo
- :C_spec          :: Float64          # coeficiente especular efectivo
- :C_diff          :: Float64          # coeficiente difuso efectivo
- :C_abs_body      :: Float64
- :C_spec_body     :: Float64
- :C_diff_body     :: Float64
- :C_abs_panel/top :: Float64
- :C_spec_panel/top:: Float64
- :C_diff_panel/top:: Float64
- :normals         :: Matrix{Float64}  # 3×N, normales de las facetas
- :areas           :: Vector{Float64}  # N, áreas de facetas
- :total_area      :: Float64          # área superficial total (numérica)
- :dims            :: NamedTuple       # dimensiones geométricas relevantes
"""
const Debris = Dict{Symbol,Any}

# -------------------- Utilidades internas --------------------

# Normal(μ,σ) truncada por arriba en 1 (como en tu MATLAB)
function _rand_trunc_normal(μ, σ; upper=1.0)
    x = μ + σ*randn()
    while x ≥ upper
        x = μ + σ*randn()
    end
    return x
end

# Cálculo de coeficientes C_abs, C_spec, C_diff a partir de medias y sigmas
function _make_optical_coeffs_body(μ_abs, σ_abs, μ_spec_aux, σ_spec_aux)
    Cabs = _rand_trunc_normal(μ_abs, σ_abs)
    Cesp_aux = _rand_trunc_normal(μ_spec_aux, σ_spec_aux)
    Cesp = Cesp_aux * (1 - Cabs)
    Cdif = 1 - Cabs - Cesp
    # Pequeños ajustes de robustez numérica
    Cdif = max(Cdif, 0.0)
    return Cabs, Cesp, Cdif
end

# Dimensiones de prisma (con y sin paneles), idénticas a tu MATLAB
function _dims_prism(sat_size::Int, sub_cat::Int)
    xdim = 0.0; ydim = 0.0; zdim = 0.0
    siz  = 0;   siz2 = 0.0

    if     sat_size == 1
        # Cubesats
        if     sub_cat == 1
            xdim = 0.1
            ydim = xdim
            zdim = xdim
            siz2 = 1.1
        elseif sub_cat == 2
            xdim = 0.1
            ydim = xdim
            zdim = 3*xdim
            siz2 = 1.2
        elseif sub_cat == 3
            xdim = 0.1
            ydim = 2*xdim
            zdim = 3*xdim
            siz2 = 1.3
        else
            xdim = 0.2
            ydim = xdim
            zdim = 1.5*xdim
            siz2 = 1.4
        end
        siz = 1
    elseif sat_size == 2
        # Sats < 100 kg
        if     sub_cat == 1
            xdim = 0.5 + 0.1*randn()
            ydim = 0.5 + 0.1*randn()
            zdim = 0.5 + 0.1*randn()
            siz2 = 2.1
        elseif sub_cat == 2
            xdim = 0.7 + 0.1*randn()
            ydim = 0.7 + 0.1*randn()
            zdim = 0.7 + 0.1*randn()
            siz2 = 2.2
        elseif sub_cat == 3
            xdim = 1.0 + 0.1*randn()
            ydim = 1.0 + 0.1*randn()
            zdim = 1.0 + 0.1*randn()
            siz2 = 2.3
        else
            xdim = 1.1
            ydim = 1.5
            zdim = 1.2
            siz2 = 2.4
        end
        siz = 2
    else
        # Grandes
        if     sub_cat == 1
            xdim = 1.25 + 0.15*randn()
            ydim = 1.25 + 0.15*randn()
            zdim = 1.25 + 0.15*randn()
            siz2 = 3.1
        elseif sub_cat == 2
            xdim = 1.5 + 0.15*randn()
            ydim = 1.5 + 0.15*randn()
            zdim = 1.5 + 0.15*randn()
            siz2 = 3.2
        elseif sub_cat == 3
            xdim = 1.8 + 0.15*randn()
            ydim = 1.8 + 0.15*randn()
            zdim = 1.8 + 0.15*randn()
            siz2 = 3.3
        else
            xdim = 4.5
            ydim = xdim
            zdim = 5.0
            siz2 = 3.4
        end
        siz = 3
    end

    return xdim, ydim, zdim, siz, siz2
end

# Dimensiones de esfera
function _dims_sphere(sat_size::Int)
    radius = 0.0
    siz = 0; siz2 = 0.0
    if     sat_size == 1
        radius = 0.1;  siz = 1; siz2 = 1.1
    elseif sat_size == 2
        radius = 0.3;  siz = 2; siz2 = 2.1
    else
        radius = 0.5;  siz = 3; siz2 = 3.1
    end
    return radius, siz, siz2
end

# Dimensiones de placa
function _dims_plate(sat_size::Int, sub_cat::Int)
    xdim = 0.0; ydim = 0.0; zdim = 0.0
    siz  = 0;   siz2 = 0.0

    if sat_size == 1
        if     sub_cat == 1
            xdim = 0.1; ydim = xdim;   zdim = 0.001; siz2 = 1.1
        elseif sub_cat == 2
            xdim = 0.2; ydim = 2*xdim; zdim = 0.002; siz2 = 1.2
        elseif sub_cat == 3
            xdim = 0.3; ydim = 3*xdim; zdim = 0.003; siz2 = 1.3
        else
            xdim = 0.1*rand(1:5)
            ydim = 4*xdim
            zdim = 0.04*xdim
            siz2 = 1.4
        end
        siz = 1
    elseif sat_size == 2
        if     sub_cat == 1
            xdim = 0.5; ydim = 2*xdim; zdim = 0.005; siz2 = 2.1
        elseif sub_cat == 2
            xdim = 0.6; ydim = 2*xdim; zdim = 0.006; siz2 = 2.2
        elseif sub_cat == 3
            xdim = 0.7; ydim = 2*xdim; zdim = 0.007; siz2 = 2.3
        else
            xdim = 0.1*rand(5:10)
            ydim = 2*xdim
            zdim = 0.08*xdim
            siz2 = 2.4
        end
        siz = 2
    else
        if     sub_cat == 1
            xdim = 1.0; ydim = 2*xdim; zdim = 0.01; siz2 = 3.1
        elseif sub_cat == 2
            xdim = 1.5; ydim = 2*xdim; zdim = 0.01; siz2 = 3.2
        elseif sub_cat == 3
            xdim = 2.0; ydim = 2*xdim; zdim = 0.02; siz2 = 3.3
        else
            xdim = 2.5; ydim = 2*xdim; zdim = 0.05; siz2 = 3.4
        end
        siz = 3
    end

    return xdim, ydim, zdim, siz, siz2
end

# Dimensiones de cilindro
function _dims_cylinder(sat_size::Int, sub_cat::Int)
    radius = 0.0; height = 0.0
    siz = 0; siz2 = 0.0

    if sat_size == 1
        if     sub_cat == 1
            radius = 0.1;  height = 0.3; siz2 = 1.1
        elseif sub_cat == 2
            radius = 0.15; height = 0.5; siz2 = 1.2
        elseif sub_cat == 3
            radius = 0.2;  height = 0.6; siz2 = 1.3
        else
            radius = 0.3;  height = 0.8; siz2 = 1.4
        end
        siz = 1
    elseif sat_size == 2
        if     sub_cat == 1
            radius = 0.3; height = 1.0; siz2 = 2.1
        elseif sub_cat == 2
            radius = 0.4; height = 1.5; siz2 = 2.2
        elseif sub_cat == 3
            radius = 0.5; height = 2.0; siz2 = 2.3
        else
            radius = 0.6; height = 2.5; siz2 = 2.4
        end
        siz = 2
    else
        if     sub_cat == 1
            radius = 0.8; height = 3.0; siz2 = 3.1
        elseif sub_cat == 2
            radius = 1.0; height = 3.5; siz2 = 3.2
        elseif sub_cat == 3
            radius = 1.2; height = 4.0; siz2 = 3.3
        else
            radius = 1.5; height = 5.0; siz2 = 3.4
        end
        siz = 3
    end

    return radius, height, siz, siz2
end

# Dimensiones de cono
function _dims_cone(sat_size::Int, sub_cat::Int)
    ratio = 0.1 * rand(1:5)  # base radius/height típico
    xdim = 0.0; ydim = 0.0; zdim = 0.0
    siz = 0;   siz2 = 0.0

    if sat_size == 1
        if     sub_cat == 1
            xdim = 0.1 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 0.2 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 0.3 + 0.05*randn()
        else
            xdim = 0.1*rand(1:5)
        end
        siz = 1; siz2 = 1.0 + 0.1*sub_cat
    elseif sat_size == 2
        if     sub_cat == 1
            xdim = 0.6 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 0.7 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 0.8 + 0.05*randn()
        else
            xdim = 0.9 + 0.05*randn()
        end
        siz = 2; siz2 = 2.0 + 0.1*sub_cat
    else
        if     sub_cat == 1
            xdim = 1.0 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 1.5 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 2.0 + 0.05*randn()
        else
            xdim = 2.5 + 0.05*randn()
        end
        siz = 3; siz2 = 3.0 + 0.1*sub_cat
    end

    ydim = xdim / ratio   # altura “efectiva”
    zdim = xdim / 2       # por analogía con tu MATLAB

    return xdim, ydim, zdim, siz, siz2
end

# Dimensiones de tronco (truncated cone)
function _dims_trunk(sat_size::Int, sub_cat::Int)
    ratio = rand(1:5)
    xdim = 0.0; ydim = 0.0; zdim = 0.0
    siz = 0;   siz2 = 0.0

    if sat_size == 1
        if     sub_cat == 1
            xdim = 0.1 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 0.2 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 0.3 + 0.05*randn()
        else
            xdim = 0.1*rand(1:5)
        end
        siz = 1; siz2 = 1.0 + 0.1*sub_cat
    elseif sat_size == 2
        if     sub_cat == 1
            xdim = 0.6 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 0.7 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 0.8 + 0.05*randn()
        else
            xdim = 0.9 + 0.05*randn()
        end
        siz = 2; siz2 = 2.0 + 0.1*sub_cat
    else
        if     sub_cat == 1
            xdim = 1.0 + 0.05*randn()
        elseif sub_cat == 2
            xdim = 1.5 + 0.05*randn()
        elseif sub_cat == 3
            xdim = 2.0 + 0.05*randn()
        else
            xdim = 2.5 + 0.05*randn()
        end
        siz = 3; siz2 = 3.0 + 0.1*sub_cat
    end

    ydim = xdim * ratio
    zdim = xdim / 2

    return xdim, ydim, zdim, siz, siz2
end

# Dimensiones de fragmento
function _dims_fragment(sat_size::Int)
    xdim = 0.0
    siz  = 0; siz2 = 0.0

    if     sat_size == 1
        xdim = 0.1; siz = 1; siz2 = 1.1
    elseif sat_size == 2
        xdim = 0.3; siz = 2; siz2 = 2.1
    else
        xdim = 0.5; siz = 3; siz2 = 3.1
    end

    return xdim, siz, siz2
end

# -------------------- API principal --------------------

"""
    generate_debris(sat_size::Int, sub_cat::Int;
                    deploy_panels::Bool=false,
                    shape_type::Int=1) -> Debris

Genera un diccionario `Debris` análogo a tu clase `Satellite` de MATLAB.

- `sat_size`    : 1 (cubesats/pequeños), 2 (<100 kg), 3 (grandes)
- `sub_cat`     : subcategoría 1..4
- `deploy_panels`: si `true`, usa “cuerpo + paneles”; si `false`, depende de `shape_type`
- `shape_type`  :
    1 = prisma
    2 = esfera
    3 = placa
    4 = cilindro
    5 = cono
    6 = tronco
    7 = fragmento

Devuelve un `Dict{Symbol,Any}` con geometría y propiedades ópticas.
"""
function generate_debris(sat_size::Int, sub_cat::Int;
                        deploy_panels::Bool=false,
                        shape_type::Int=1)

    debris = Debris()

    debris[:sat_size]      = sat_size
    debris[:sub_cat]       = sub_cat
    debris[:deploy_panels] = deploy_panels
    debris[:shape_type]    = shape_type

    # Exponente Phong global (alpha)
    debris[:alpha] = 1.2 + 0.1*randn()

    # Paneles (siempre los modelamos como “material solar panel”)
    Cabs_panel, Cesp_panel, Cdif_panel = _make_optical_coeffs_body(0.75, 0.1, 0.8, 0.05)

    if deploy_panels
        # Cuerpo + paneles desplegados (prisma)
        Cabs_body, Cesp_body, Cdif_body = _make_optical_coeffs_body(0.62, 0.12, 0.7, 0.1)

        xdim, ydim, zdim, siz, siz2 = _dims_prism(sat_size, sub_cat)
        normals, areas, total_area  = generateRandomShape(1, xdim, ydim, zdim)  # cubo / prisma :contentReference[oaicite:2]{index=2}

        debris[:shape_name] = :prism_with_panels
        debris[:siz]        = siz
        debris[:siz2]       = siz2
        debris[:normals]    = normals      # 3×N
        debris[:areas]      = areas
        debris[:total_area] = total_area
        debris[:dims]       = (xdim=xdim, ydim=ydim, zdim=zdim)

        # División geométrica ideal: paneles = caras superior+inferior, resto cuerpo
        A_panels = 2 * xdim * ydim
        A_body   = 2 * (xdim*zdim + ydim*zdim)
        A_geom   = A_panels + A_body

        C_spec_eff = (Cesp_panel*A_panels + Cesp_body*A_body) / A_geom
        C_diff_eff = (Cdif_panel*A_panels + Cdif_body*A_body) / A_geom
        C_abs_eff  = 1 - C_spec_eff - C_diff_eff

        debris[:C_abs_body]   = Cabs_body
        debris[:C_spec_body]  = Cesp_body
        debris[:C_diff_body]  = Cdif_body
        debris[:C_abs_panel]  = Cabs_panel
        debris[:C_spec_panel] = Cesp_panel
        debris[:C_diff_panel] = Cdif_panel

        debris[:C_abs]  = C_abs_eff
        debris[:C_spec] = C_spec_eff
        debris[:C_diff] = C_diff_eff

    else
        # Sin paneles desplegados
        # Material del cuerpo
        Cabs_body, Cesp_body, Cdif_body = _make_optical_coeffs_body(0.62, 0.12, 0.7, 0.1)

        if shape_type == 1
            # Prisma sin paneles (cuerpo + tapas "top")
            Cabs_top, Cesp_top, Cdif_top = _make_optical_coeffs_body(0.55, 0.2, 0.7, 0.1)

            xdim, ydim, zdim, siz, siz2 = _dims_prism(sat_size, sub_cat)
            normals, areas, total_area  = generateRandomShape(1, xdim, ydim, zdim)

            debris[:shape_name] = :prism_no_panels
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (xdim=xdim, ydim=ydim, zdim=zdim)

            A_top   = 2 * xdim * ydim
            A_body  = 2 * (xdim*zdim + ydim*zdim)
            A_geom  = A_top + A_body

            C_spec_eff = (Cesp_top*A_top + Cesp_body*A_body) / A_geom
            C_diff_eff = (Cdif_top*A_top + Cdif_body*A_body) / A_geom
            C_abs_eff  = 1 - C_spec_eff - C_diff_eff

            debris[:C_abs_body]  = Cabs_body
            debris[:C_spec_body] = Cesp_body
            debris[:C_diff_body] = Cdif_body
            debris[:C_abs_top]   = Cabs_top
            debris[:C_spec_top]  = Cesp_top
            debris[:C_diff_top]  = Cdif_top

            debris[:C_abs]  = C_abs_eff
            debris[:C_spec] = C_spec_eff
            debris[:C_diff] = C_diff_eff

        elseif shape_type == 2
            # Esfera homogénea
            radius, siz, siz2 = _dims_sphere(sat_size)
            normals, areas, total_area = generateRandomShape(2, radius, 0.0, 0.0)

            debris[:shape_name] = :sphere
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (radius=radius,)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_body
            debris[:C_diff] = Cdif_body

        elseif shape_type == 3
            # Placa (panel, homogénea)
            xdim, ydim, zdim, siz, siz2 = _dims_plate(sat_size, sub_cat)
            normals, areas, total_area  = generateRandomShape(3, xdim, ydim, zdim)

            debris[:shape_name] = :plate
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (xdim=xdim, ydim=ydim, zdim=zdim)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_panel   # parecido a “panel solar”
            debris[:C_diff] = Cdif_panel

        elseif shape_type == 4
            # Cilindro
            radius, height, siz, siz2 = _dims_cylinder(sat_size, sub_cat)
            normals, areas, total_area = generateRandomShape(4, radius, height, 0.0)

            debris[:shape_name] = :cylinder
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (radius=radius, height=height)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_body
            debris[:C_diff] = Cdif_body

        elseif shape_type == 5
            # Cono
            xdim, ydim, zdim, siz, siz2 = _dims_cone(sat_size, sub_cat)
            normals, areas, total_area  = generateRandomShape(5, xdim, ydim, zdim)

            debris[:shape_name] = :cone
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (xdim=xdim, ydim=ydim, zdim=zdim)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_body
            debris[:C_diff] = Cdif_body

        elseif shape_type == 6
            # Tronco de cono
            xdim, ydim, zdim, siz, siz2 = _dims_trunk(sat_size, sub_cat)
            normals, areas, total_area  = generateRandomShape(6, xdim, ydim, zdim)

            debris[:shape_name] = :trunk
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (xdim=xdim, ydim=ydim, zdim=zdim)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_body
            debris[:C_diff] = Cdif_body

        elseif shape_type == 7
            # Fragmento aleatorio
            xdim, siz, siz2 = _dims_fragment(sat_size)
            normals, areas, total_area = generateRandomShape(7, xdim, 0.0, 0.0)

            debris[:shape_name] = :fragment
            debris[:siz]        = siz
            debris[:siz2]       = siz2
            debris[:normals]    = normals
            debris[:areas]      = areas
            debris[:total_area] = total_area
            debris[:dims]       = (char_length=xdim,)

            debris[:C_abs]  = Cabs_body
            debris[:C_spec] = Cesp_body
            debris[:C_diff] = Cdif_body

        else
            error("shape_type debe estar entre 1 y 7.")
        end
    end

    return debris
end

end # module
