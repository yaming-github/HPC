import Dates
import LinearAlgebra.BLAS

vectorN = 1000000
matrixN = 512

vectorA = Array{Float64, 1}(undef, vectorN)
vectorB = Array{Float64, 1}(undef, vectorN)
vectorC = Array{Float64, 1}(undef, vectorN)
matrixA = Array{Float64, 2}(undef, matrixN, matrixN)
matrixB = Array{Float64, 2}(undef, matrixN, matrixN)
matrixC = Array{Float64, 2}(undef, matrixN, matrixN)

for i = 1:vectorN
    vectorB[i] = sin(i) * sin(i)
    vectorC[i] = cos(i) * cos(i)
end

start = Dates.now()
for i = 1:vectorN
    vectorA[i] = vectorB[i] + vectorB[i]
end
duration = Dates.now() - start
sum = 0.0
for i = 1:vectorN
    global sum += vectorA[i]
end
println("JULIA VECTOR: ", sum / vectorN)
println("JULIA VECTOR: ", duration)
for i = 1:vectorN
    vectorA[i] = 0
end

start = Dates.now()
Threads.@threads for i = 1:vectorN
    vectorA[i] = vectorB[i] + vectorB[i]
end
duration = Dates.now() - start
sum = 0.0
for i = 1:vectorN
    global sum += vectorA[i]
end
println("JULIA VECTOR threads: ", sum / vectorN)
println("JULIA VECTOR threads: ", duration)

for i = 1:matrixN
    for j = 1:matrixN
        matrixB[i, j] = i + j
        matrixC[i, j] = i - j
    end
end

start = Dates.now()
for i = 1:matrixN
    for j = 1:matrixN
        tmp = 0.0
        for k = 1:matrixN
            tmp += matrixB[i,k] * matrixC[k,j]
        end
        matrixA[i,j] = tmp
    end
end
duration = Dates.now() - start
println("JULIA MATMUL: ", matrixA[6, 9])
println("JULIA MATMUL: ", duration)

start = Dates.now()
matrixA = matrixB * matrixC
duration = Dates.now() - start

println("JULIA MATMUL LA: ", matrixA[6, 9])
println("JULIA MATMUL LA: ", duration)

start = Dates.now()
matrixA = BLAS.gemm('N', 'N', matrixB, matrixC)
duration = Dates.now() - start

println("JULIA MATMUL BLAS: ", matrixA[6, 9])
println("JULIA MATMUL BLAS: ", duration)
