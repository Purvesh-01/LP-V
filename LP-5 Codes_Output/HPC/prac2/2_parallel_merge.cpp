#include <iostream>
#include <omp.h>
#include <iomanip>

using namespace std;

// 🔹 Print function
void printArray(int arr[], int n)
{
    cout << "Sorted array (first 20 elements): ";
    for (int i = 0; i < min(n, 20); i++)
    {
        cout << arr[i] << " ";
    }
    cout << "...\n";
}

// Merge function
void merge(int arr[], int low, int mid, int high)
{
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int *left = new int[n1];
    int *right = new int[n2];

    for (int i = 0; i < n1; i++)
        left[i] = arr[low + i];
    for (int j = 0; j < n2; j++)
        right[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }

    while (i < n1)
        arr[k++] = left[i++];
    while (j < n2)
        arr[k++] = right[j++];

    delete[] left;
    delete[] right;
}

// Sequential Merge Sort
void mergeSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

// Parallel Merge Sort (Task-based)
void parallelMergeSort(int arr[], int low, int high)
{
    if (low < high)
    {

        // Cutoff to avoid tiny tasks
        if (high - low < 1000)
        {
            mergeSort(arr, low, high);
            return;
        }

        int mid = (low + high) / 2;

#pragma omp task shared(arr)
        parallelMergeSort(arr, low, mid);

#pragma omp task shared(arr)
        parallelMergeSort(arr, mid + 1, high);

#pragma omp taskwait
        merge(arr, low, mid, high);
    }
}

int main()
{
    int n = 1000000;
    int *arr = new int[n];

    double start, end;

    cout << fixed << setprecision(9);

    // Initialize array
    for (int i = 0; i < n; i++)
        arr[i] = n - i;

    // 🔹 Sequential
    start = omp_get_wtime();
    mergeSort(arr, 0, n - 1);
    end = omp_get_wtime();

    cout << "Sequential Time: " << (end - start) << " sec\n";
    printArray(arr, n);

    // Reset array
    for (int i = 0; i < n; i++)
        arr[i] = n - i;

    // 🔹 Parallel
    start = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp single
        parallelMergeSort(arr, 0, n - 1);
    }

    end = omp_get_wtime();

    cout << "Parallel Time:   " << (end - start) << " sec\n";
    printArray(arr, n);
    delete[] arr;
    return 0;
}