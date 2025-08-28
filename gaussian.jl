# Cris' function rewritten
# Creates a half Gaussian kernel with the given parameters.
function MakeHalfGaussian(sigma, derivativeOrder)
    # Compute the size of the kernel and adjust it if necessary
    halfFilterSize = 1 + 9

    # Initialize the filter array and some variables
    filter = zeros(halfFilterSize)
    r0 = halfFilterSize
    sigma2 = sigma * sigma

    # Compute the filter values based on the derivative order
    # ignores constant factors
    # does not cycle over central values
    if derivativeOrder == 0
        factor = -0.5 / sigma2
        normalization = 0.0
        filter[r0] = 1.0  # central value is 1 since exp(0) = 1
        for rr in 1:halfFilterSize-1
            rad = Float64(rr)
            g = exp(factor * rad^2)
            filter[r0-rr] = g
            normalization += g
        end
        normalization = 1.0 / (normalization * 2 + 1) # central value added, and account for both tails
        for rr in 1:halfFilterSize
            filter[rr] *= normalization
        end

    elseif derivativeOrder == 1
        factor = -0.5 / sigma2
        moment = 0.0
        filter[r0] = 0.0  # central value must be 0 since x as prefactor in 1st derivative
        for rr in 1:halfFilterSize-1
            rad = Float64(rr)
            g = rad * exp(factor * rad^2)
            filter[r0-rr] = g
            moment += rad * g
        end
        normalization = 1.0 / (2.0 * moment)  # moment in centre is 0
        for rr in 1:halfFilterSize-1 # centre needs no normalisation since it is 0
            filter[rr] *= normalization
        end

    elseif derivativeOrder == 2
        norm = 1.0 / (sqrt(2.0 * pi) * sigma * sigma2)
        mean = 0.0
        filter[r0] = -norm
        for rr in 1:halfFilterSize-1
            rad = Float64(rr)
            sr2 = rad^2 / sigma2
            g = (sr2 - 1.0) * norm * exp(-0.5 * sr2)
            filter[r0-rr] = g
            mean += g
        end
        mean = (mean * 2.0 + filter[r0]) / (Float64(r0) * 2.0 + 1.0)
        filter[r0] -= mean
        moment = 0.0
        for rr in 1:halfFilterSize-1
            rad = Float64(rr)
            filter[r0-rr] -= mean
            moment += rad^2 * filter[r0-rr]
        end
        normalization = 1.0 / moment
        for rr in 1:halfFilterSize
            filter[rr] *= normalization
        end
    elseif derivativeOrder == 3
        norm = 1.0 / (sqrt(2.0 * pi) * sigma * sigma2^2)
        filter[r0] = 0.0
        moment = 0.0
        for rr in 1:halfFilterSize-1
            rad = Float64(rr)
            rr2 = rad^2
            sr2 = rr2 / sigma2
            g = norm * exp(-0.5 * sr2) * (rad * (3.0 - sr2))
            filter[r0-rr] = g
            moment += g * rr2 * rad
        end
        normalization = 3.0 / moment
        for rr in 0:halfFilterSize-1
            filter[rr] *= normalization
        end
    else
        error("Not implemented")
    end
    
    # Return the filter array
    return filter
end

function makeGaussJH(sigma,width)
    N(x) = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma))
    dN(x) = -(x/sigma^2) * N(x)
    ddN(x) = -(1/sigma^2) * N(x) - (x/sigma^2) * dN(x)

    #x = collect(-width:width)
    x = ImageFiltering.OffsetVector(-width:width,-(width+1))
    #ddg = ddN.(x)
    g = KernelFactors.gaussian(sigma,2*width+1)
    g_dd = g .* ((x.^2 .- sigma^2) ./ (sigma^4))
    
    E = mean(g_dd)
    g_dd .= g_dd .- E

    m2 = dot(g_dd[-width:-1],(-width:-1).^2)
    g_dd .= g_dd ./ (m2)

    return g_dd
end


function makeGauss2(sigma,width)
    N(x) = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma))
    dN(x) = -(x/sigma^2) * N(x)
    ddN(x) = -(1/sigma^2) * N(x) - (x/sigma^2) * dN(x)

    x = collect(-width:width)
    g_dd = ddN.(x)
    g_dd = ImageFiltering.OffsetVector(g_dd,-(width+1))
    
    E = mean(g_dd)
    g_dd .= g_dd .- E  # assure zero result in convolution of constant

    #m2 = dot(g_dd[-width:-1],(-width:-1).^2)
    #g_dd .= g_dd ./ (m2)
    m2 = dot(g_dd,x.^2)
    g_dd .= 2*g_dd ./ (m2)

    return g_dd
end

dd = makeGaussJH(3,9)
w = div(length(dd), 2)
x = collect(-w:w)

dot(dd,ones(length(dd)))
dot(dd,x)
dot(dd,x.^2)