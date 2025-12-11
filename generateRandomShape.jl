using LinearAlgebra
using QHull

"""
    generateRandomShape(shapeType, xdim, ydim, zdim)

Devuelve `(normals, areas, tot_area)` de un sólido triangulado.

shapeType:
1 = Cubo/paralelepípedo (dims xdim, ydim, zdim)
2 = Esfera (radio = xdim; malla 20×20)
3 = Placa (paralelepípedo delgado con esas dims)
4 = Cilindro macizo (radio = xdim, altura = ydim)
5 = Cono hueco
6 = Cono truncado hueco
7 = Fragmento aleatorio (envolvente convexa; requiere QHull.jl)

Retorno:
normals :: Matrix{Float64}  (nfaces × 3)
areas   :: Vector{Float64}  (nfaces)
tot_area:: Float64
"""
function generateRandomShape(shapeType::Int, xdim::Real, ydim::Real, zdim::Real)
    # Inicialización segura (0×3 y 0×3)
    verts = Array{Float64}(undef, 0, 3)
    tris  = Array{Int}(undef, 0, 3)

    if shapeType == 1
        verts, tris = cube_vertices_faces(xdim, ydim, zdim)
    elseif shapeType == 2
        verts, tris = sphere_mesh(xdim, 20, 20)
    elseif shapeType == 3
        verts, tris = cube_vertices_faces(xdim, ydim, zdim)
    elseif shapeType == 4
        verts, tris = cylinder_mesh(xdim, ydim; Nt=20, Nz=1)
    elseif shapeType == 5
        verts, tris = hollow_cone(xdim, ydim, false)
    elseif shapeType == 6
        verts, tris = hollow_cone(xdim, ydim, true)
    elseif shapeType == 7
        verts, tris = random_fragment(xdim)  # si no usas QHull, reemplaza esta función
    else
        error("shapeType fuera de rango (1..7).")
    end

    normals, areas, tot_area = calculateNormals(verts, tris)
    return normals, areas, tot_area
end


# ---------- Helpers de malla ----------

# Paralelepípedo centrado en el origen
function cube_vertices_faces(xd, yd, zd)
    x = xd/2; y = yd/2; z = zd/2
    V = [
        -x -y -z;
        x -y -z;
        x  y -z;
        -x  y -z;
        -x -y  z;
        x -y  z;
        x  y  z;
        -x  y  z
    ]
    F = [
        1 2 3; 1 3 4;
        5 6 7; 5 7 8;
        1 2 6; 1 6 5;
        2 3 7; 2 7 6;
        3 4 8; 3 8 7;
        4 1 5; 4 5 8
    ]
    return Matrix{Float64}(V), Matrix{Int}(F)
end

# Malla de esfera (grid lon-lat)
function sphere_mesh(radius::Real, nθ::Int, nφ::Int)
    θs = range(0, 2π, length=nθ+1)[1:end-1]
    φs = range(-π/2, π/2, length=nφ)

    points = Vector{NTuple{3,Float64}}()
    for φ in φs, θ in θs
        x = radius*cos(φ)*cos(θ)
        y = radius*cos(φ)*sin(θ)
        z = radius*sin(φ)
        push!(points, (x,y,z))
    end
    V = reduce(vcat, (collect(p)' for p in points))

    F = Int[]
    idx = (i,j)-> (i-1)*length(θs) + j
    for i in 1:(length(φs)-1)
        for j in 1:length(θs)
            j2 = (j % length(θs)) + 1
            v1 = idx(i, j)
            v2 = idx(i, j2)
            v3 = idx(i+1, j)
            v4 = idx(i+1, j2)
            append!(F, (v1, v2, v3, v2, v4, v3))
        end
    end
    F = reshape(F, 3, :)'
    return V, F
end

# Cilindro macizo
function cylinder_mesh(R::Real, H::Real; Nt::Int=20, Nz::Int=1)
    θs = range(0, 2π, length=Nt+1)[1:end-1]
    zs = range(-H/2, H/2, length=Nz+2)

    verts = Vector{Tuple{Float64,Float64,Float64}}()
    for z in zs
        for θ in θs
            push!(verts, (R*cos(θ), R*sin(θ), z))
        end
    end
    push!(verts, (0.0, 0.0, -H/2))
    push!(verts, (0.0, 0.0,  H/2))

    V = reduce(vcat, (collect(p)' for p in verts))
    nring = length(θs)
    nrings = length(zs)
    idx = (ring, j)-> (ring-1)*nring + j
    bottom_center = size(V,1)-1
    top_center    = size(V,1)
    F = Int[]

    for r in 1:(nrings-1)
        for j in 1:nring
            j2 = (j % nring) + 1
            v1 = idx(r, j)
            v2 = idx(r, j2)
            v3 = idx(r+1, j)
            v4 = idx(r+1, j2)
            append!(F, (v1,v2,v3,  v2,v4,v3))
        end
    end
    for j in 1:nring
        j2 = (j % nring) + 1
        append!(F, (bottom_center, idx(1,j2), idx(1,j)))
    end
    for j in 1:nring
        j2 = (j % nring) + 1
        append!(F, (top_center, idx(nrings,j), idx(nrings,j2)))
    end
    F = reshape(F, 3, :)'
    return V, F
end

# Cono hueco o truncado
function hollow_cone(base_radius::Real, height::Real, is_truncated::Bool)
    n = 50
    θ = range(0, 2π, length=n+1)[1:end-1]

    if is_truncated
        expansion_ratio = rand(2:10)
        top_radius = base_radius / expansion_ratio
        X = vcat(base_radius .* cos.(θ), top_radius .* cos.(θ))
        Y = vcat(base_radius .* sin.(θ), top_radius .* sin.(θ))
        Z = vcat(zeros(n),              fill(height, n))

        Fext = Int[]
        for i in 1:n
            i2 = (i % n) + 1
            append!(Fext, ( i, i2, n+i2,   i, n+i2, n+i ))
        end
        Fint = Int[]
        for i in 1:n
            i2 = (i % n) + 1
            append!(Fint, ( n+i2, n+i, i,   n+i2, i, i2 ))
        end
        V = hcat(X, Y, Z)
        F = reshape(vcat(Fext, Fint), 3, :)'
        return V, F
    else
        X = vcat(base_radius .* cos.(θ), 0.0)
        Y = vcat(base_radius .* sin.(θ), 0.0)
        Z = vcat(zeros(n),               height)
        apex = n + 1
        Fext = Int[]
        Fint = Int[]
        for i in 1:n
            i2 = (i % n) + 1
            append!(Fext, (i, i2, apex))
            append!(Fint, (apex, i, i2))
        end
        V = hcat(X, Y, Z)
        F = reshape(vcat(Fext, Fint), 3, :)'
        return V, F
    end
end

# ---------- Normales y áreas ----------
function calculateNormals(V::AbstractMatrix{<:Real}, F::AbstractMatrix{<:Integer})
    M = size(F,1)
    normals = zeros(Float64, 3, M)
    areas   = zeros(Float64, M)
    center  = vec(mean(V, dims=1))

    for i in 1:M
        v1, v2, v3 = F[i,1], F[i,2], F[i,3]
        p1 = @view V[v1, :]
        p2 = @view V[v2, :]
        p3 = @view V[v3, :]
        e1 = p2 .- p1
        e2 = p3 .- p1
        n = [
            e1[2]*e2[3] - e1[3]*e2[2],
            e1[3]*e2[1] - e1[1]*e2[3],
            e1[1]*e2[2] - e1[2]*e2[1]
        ]
        Atri = 0.5 * norm(n)
        fc = (p1 .+ p2 .+ p3) ./ 3
        if dot(n, fc .- center) < 0
            n .= -n
        end
        normals[:, i] .= n ./ norm(n)
        areas[i] = Atri
    end
    return normals, areas, sum(areas)
end

# ---------- Fragmento aleatorio ----------
function random_fragment(obj_size::Real)
    num_points = rand(10:30)
    pts = generate_points_in_sphere(num_points)
    scale_factor = ((3*obj_size) / (4π))^(1/3)
    pts .= pts .* scale_factor
    hull = qhull(pts', "Qt")
    V = pts
    F = permutedims(hull.facets, (2,1))
    return Matrix{Float64}(V), Matrix{Int}(F)
end

function generate_points_in_sphere(num_points::Int)
    pts = Float64[]
    while size(reshape(pts, :, 3), 1) < num_points
        cand = 2 .* rand(num_points,3) .- 1
        inside = vec(sqrt.(sum(cand.^2, dims=2))) .<= 1
        cand = cand[inside, :]
        pts = vcat(pts, vec(cand'))
    end
    P = reshape(pts, 3, :)'
    return P[1:num_points, :]
end
