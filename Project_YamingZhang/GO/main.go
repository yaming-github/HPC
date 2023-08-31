package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

const (
	vectorN   = 1000000
	matrixN   = 512
	threadNum = 10
)

var vectorA [vectorN]float64
var matrixA [matrixN][matrixN]float64

func main() {
	var vectorB, vectorC [vectorN]float64
	for i := 0; i < vectorN; i++ {
		vectorB[i] = math.Sin(float64(i)) * math.Sin(float64(i))
		vectorC[i] = math.Cos(float64(i)) * math.Cos(float64(i))
	}
	start := time.Now()
	for i := 0; i < vectorN; i++ {
		vectorA[i] = vectorB[i] + vectorC[i]
	}
	duration := time.Since(start).Milliseconds()
	sum := 0.0
	for i := 0; i < vectorN; i++ {
		sum += vectorA[i]
	}
	fmt.Printf("GOLANG VECTOR: %f\n", sum/vectorN)
	fmt.Printf("GOLANG VECTOR: %d ms\n", duration)

	numPerThread := vectorN / threadNum
	wg := sync.WaitGroup{}
	start = time.Now()
	for i := 0; i < threadNum; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < numPerThread; j++ {
				vectorA[i*numPerThread+j] = vectorB[i*numPerThread+j] + vectorC[i*numPerThread+j]
			}
		}(i)
	}
	wg.Wait()
	duration = time.Since(start).Milliseconds()
	sum = 0.0
	for i := 0; i < vectorN; i++ {
		sum += vectorA[i]
	}
	fmt.Printf("GOLANG VECTOR GOROUTINE: %f\n", sum/vectorN)
	fmt.Printf("GOLANG VECTOR GOROUTINE: %d ms\n", duration)

	var matrixB, matrixC [matrixN][matrixN]float64
	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			matrixB[i][j] = float64(i + j)
			matrixC[i][j] = float64(i - j)
		}
	}

	start = time.Now()
	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			tmp := 0.0
			for k := 0; k < matrixN; k++ {
				tmp += matrixB[i][k] * matrixC[k][j]
			}
			matrixA[i][j] = tmp
		}
	}
	duration = time.Since(start).Milliseconds()
	fmt.Printf("GOLANG MATMUL: %f\n", matrixA[7][8])
	fmt.Printf("GOLANG MATMUL: %d ms\n", duration)
	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			matrixA[i][j] = 0.0
		}
	}

	start = time.Now()
	for i := 0; i < matrixN; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < matrixN; j++ {
				tmp := 0.0
				for k := 0; k < matrixN; k++ {
					tmp += matrixB[i][k] * matrixC[k][j]
				}
				matrixA[i][j] = tmp
			}
		}(i)
	}
	wg.Wait()
	duration = time.Since(start).Milliseconds()
	fmt.Printf("GOLANG MATMUL GOROUTNE ROW: %f\n", matrixA[7][8])
	fmt.Printf("GOLANG MATMUL GOROUTNE ROW: %d ms\n", duration)

	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			matrixA[i][j] = 0.0
		}
	}

	workerLimit := make(chan struct{}, runtime.GOMAXPROCS(0))

	wg.Add(matrixN * matrixN)
	start = time.Now()
	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			workerLimit <- struct{}{}
			go func(i, j int) {
				defer func() {
					wg.Done()
					<-workerLimit
				}()
				tmp := 0.0
				for k := 0; k < matrixN; k++ {
					tmp += matrixB[i][k] * matrixC[k][j]
				}
				matrixA[i][j] = tmp
			}(i, j)
		}
	}
	wg.Wait()
	duration = time.Since(start).Milliseconds()
	fmt.Printf("GOLANG MATMUL GOROUTNE POINT: %f\n", matrixA[7][8])
	fmt.Printf("GOLANG MATMUL GOROUTNE POINT: %d ms\n", duration)

	blasBData := make([]float64, matrixN*matrixN)
	blasCData := make([]float64, matrixN*matrixN)
	for i := 0; i < matrixN; i++ {
		for j := 0; j < matrixN; j++ {
			blasBData[i*matrixN+j] = float64(i + j)
			blasCData[i*matrixN+j] = float64(i - j)
		}
	}
	blasB := blas64.General{
		Rows:   matrixN,
		Cols:   matrixN,
		Data:   blasBData,
		Stride: matrixN,
	}
	blasC := blas64.General{
		Rows:   matrixN,
		Cols:   matrixN,
		Data:   blasCData,
		Stride: matrixN,
	}
	blasA := blas64.General{
		Rows:   matrixN,
		Cols:   matrixN,
		Data:   make([]float64, matrixN*matrixN),
		Stride: matrixN,
	}
	start = time.Now()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blasB, blasC, 0, blasA)
	duration = time.Since(start).Milliseconds()
	fmt.Printf("GOLANG MATMUL BLAS: %f\n", blasA.Data[7*matrixN+8])
	fmt.Printf("GOLANG MATMUL BLAS: %d ms\n", duration)
}
