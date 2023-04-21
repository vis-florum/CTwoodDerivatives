using Pkg
Pkg.activate("./src")
Pkg.instantiate()

try
    using Images
    using Plots
    using PyCall
    using LinearAlgebra
    using Statistics
catch
    "activate environment?"
end

pic = load("images/circle.png")

gry = channelview(pic)
gry = Float32.(gry)
gry = gry[1,:,:]

dip = pyimport("diplib")
EV = dip.StructureTensor(gry)
outs = ["l1", "l2", "orientation", "energy", "anisotropy1", "anisotropy2", "curvature"]
EVA = dip.StructureTensorAnalysis(EV,outs)

py"""
def GST(I):
    import diplib as dip
    EV = dip.StructureTensor(I)
    outs = ["l1", "l2", "orientation", "energy", "anisotropy1", "anisotropy2", "curvature"]
    EVA = dip.StructureTensorAnalysis(EV,outs)
    return EVA
"""
EVA = py"GST"(gry)

# plot a montage of all fields in EVA as images
montage = [EVA[1], EVA[2], EVA[3], EVA[4], EVA[5], EVA[6], EVA[7]]
plot(montage..., layout=(1,7), size=(1000,1000))

# Set up the plot grid layout and size
montage_layout = grid(2, 4) # 2 rows, 4 columns
montage_size = (1000, 500)  # Width and height in pixels

# Create an empty plot with the specified layout and size
montage_plot = plot(layout = montage_layout, size = montage_size)

# Add each image to the plot
EVA_images = [Gray.(img) for img in EVA]
for (index, img) in enumerate(EVA_images)
    plot!(montage_plot[index], img, axis = false)
end

# Display the montage
montage_plot


# plot a vector field
x = collect(1:size(gry,1))
x = x[1:10:end]
y = collect(1:size(gry,2))
y = y[1:10:end]
X = x'.*ones(length(x))
Y = y.*ones(length(y))'

u = cos(EVA[3])
v = sin(EVA[3])
U = u[1:10:end, 1:10:end]
V = v[1:10:end, 1:10:end]
plot(Gray.(gry))
quiver!(X,Y, quiver=(u, v))

# rewrite x, y = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0])) to julia
x, y = Iterators.product(0:size(EV, 2)-1, 0:size(EV, 1)-1) |> Tuple
#|> vec

