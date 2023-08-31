import math
import threading
import time

import numpy
import torch

vectorN = 1000000
matrixN = 512
threadNum = 10

vectorA = torch.zeros((vectorN,))
matrixA = torch.zeros((matrixN, matrixN))


def vectorFun(b, c, i, num_per_thread):
    for j in range(num_per_thread):
        vectorA[i * num_per_thread + j] = b[i * num_per_thread + j] + c[i * num_per_thread + j]


if __name__ == '__main__':
    vectorB = torch.zeros((vectorN,))
    vectorC = torch.zeros((vectorN,))
    for i in range(vectorN):
        vectorB[i] = math.sin(i) * math.sin(i)
        vectorC[i] = math.cos(i) * math.cos(i)
    start = time.time_ns()
    for i in range(vectorN):
        vectorA[i] = vectorB[i] + vectorC[i]
    duration = (time.time_ns() - start) / 1000000
    sumTmp = 0.0
    for i in range(vectorN):
        sumTmp += vectorA[i]
    print("PYTHON VECTOR: {}".format(sumTmp / vectorN))
    print("PYTHON VECTOR: {} ms".format(duration))
    for i in range(vectorN):
        vectorA[i] = 0.0


    numPerThread = vectorN // threadNum
    start = time.time_ns()
    for i in range(threadNum):
        t = threading.Thread(target=vectorFun, args=(vectorB, vectorC, i, numPerThread))
        t.start()
        t.join()
    duration = (time.time_ns() - start) / 1000000
    sumTmp = 0.0
    for i in range(vectorN):
        sumTmp += vectorA[i]
    print("PYTHON VECTOR ADD THREADING: {}".format(sumTmp / vectorN))
    print("PYTHON VECTOR ADD THREADING: {} ms".format(duration))
    for i in range(vectorN):
        vectorA[i] = 0.0

    start = time.time_ns()
    vectorA = vectorB + vectorC
    duration = (time.time_ns() - start) / 1000000
    sumTmp = 0.0
    for i in range(vectorN):
        sumTmp += vectorA[i]
    print("PYTHON VECTOR PYTORCH: {}".format(sumTmp / vectorN))
    print("PYTHON VECTOR PYTORCH: {} ms".format(duration))

    matrixB = torch.zeros((matrixN, matrixN))
    matrixC = torch.zeros((matrixN, matrixN))
    for i in range(matrixN):
        for j in range(matrixN):
            matrixB[i][j] = float(i + j)
            matrixC[i][j] = float(i - j)

    start = time.time_ns()
    for i in range(matrixN):
        for j in range(matrixN):
            tmp = 0.0
            for k in range(matrixN):
                tmp += matrixB[i][k] * matrixC[k][j]
            matrixA[i][j] = tmp
    duration = (time.time_ns() - start) / 1000000
    print("PYTHON MATMUL: {}".format(matrixA[7][8]))
    print("PYTHON MATMUL: {} ms".format(duration))

    start = time.time_ns()
    matrixA = torch.matmul(matrixB, matrixC)
    duration = (time.time_ns() - start) / 1000000
    print("PYTHON MATMUL TORCH: {}".format(matrixA[7][8]))
    print("PYTHON MATMUL TORCH: {} ms".format(duration))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        matrixB_gpu = matrixB.to(device)
        matrixC_gpu = matrixC.to(device)
        torch.cuda.synchronize()
        start = time.time_ns()
        matrixA = torch.matmul(matrixB_gpu, matrixC_gpu)
        torch.cuda.synchronize()
        duration = (time.time_ns() - start) / 1000000
        print("PYTHON MATMUL CUDA: {}".format(matrixA[7][8]))
        print("PYTHON MATMUL CUDA: {} ms".format(duration))
