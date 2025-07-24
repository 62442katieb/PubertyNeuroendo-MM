import numpy as np
import os
import sys

def read_npy_file(file_path):
    """
    Read and display information about a .npy file
    
    Args:
        file_path (str): Path to the .npy file
    """
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)
        
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Size: {data.size} elements")
        
        # Show sample data based on array dimensions
        if data.ndim == 0:
            # Scalar
            print(f"Value: {data}")
        elif data.ndim == 1:
            # 1D array
            print(f"First 10 elements: {data[:10]}")
            if len(data) > 10:
                print(f"Last 5 elements: {data[-5:]}")
        elif data.ndim == 2:
            # 2D array
            print(f"First 5 rows, first 5 columns:")
            print(data[:5, :5])
            if data.shape[0] > 5 or data.shape[1] > 5:
                print(f"Shape summary: {data.shape[0]} rows × {data.shape[1]} columns")
        elif data.ndim == 3:
            # 3D array (like prediction records)
            print(f"3D array: {data.shape[0]} × {data.shape[1]} × {data.shape[2]}")
            print(f"First slice [0,:,:]:")
            print(data[0, :5, :])
        else:
            # Higher dimensional
            print(f"Higher dimensional array: {data.shape}")
            print(f"First few elements: {data.flat[:10]}")
            
        # Show statistics for numeric data
        if np.issubdtype(data.dtype, np.number):
            print(f"\nStatistics:")
            print(f"  Mean: {np.nanmean(data):.4f}")
            print(f"  Std: {np.nanstd(data):.4f}")
            print(f"  Min: {np.nanmin(data):.4f}")
            print(f"  Max: {np.nanmax(data):.4f}")
            if np.any(np.isnan(data)):
                print(f"  NaN count: {np.sum(np.isnan(data))}")
                
        return data
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def list_npy_files(directory):
    """
    List all .npy files in a directory
    
    Args:
        directory (str): Directory path to search
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found")
        return []
        
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    if npy_files:
        print(f"Found {len(npy_files)} .npy files in '{directory}':")
        for i, file in enumerate(sorted(npy_files), 1):
            print(f"  {i}. {file}")
    else:
        print(f"No .npy files found in '{directory}'")
    
    return sorted(npy_files)

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_npy.py <file_path>")
        print("   or: python read_npy.py list <directory>")
        print("\nExamples:")
        print("  python read_npy.py ./train_output/logs/dnn_val_cor_record.npy")
        print("  python read_npy.py list ./train_output/logs")
        return
    
    if sys.argv[1] == "list":
        directory = sys.argv[2] if len(sys.argv) > 2 else "./train_output/logs"
        list_npy_files(directory)
    else:
        file_path = sys.argv[1]
        data = read_npy_file(file_path)
        
        # Interactive mode - ask if user wants to see more
        if data is not None:
            while True:
                print("\nOptions:")
                print("1. Show full array")
                print("2. Show specific slice/index")
                print("3. Save as text file")
                print("4. Exit")
                
                choice = input("Enter choice (1-4): ").strip()
                
                if choice == "1":
                    print("\nFull array:")
                    print(data)
                elif choice == "2":
                    slice_str = input("Enter slice (e.g., '0', ':5', '0,:,0'): ").strip()
                    try:
                        result = eval(f"data[{slice_str}]")
                        print(f"\ndata[{slice_str}]:")
                        print(result)
                    except Exception as e:
                        print(f"Invalid slice: {e}")
                elif choice == "3":
                    output_file = input("Enter output filename (e.g., 'output.txt'): ").strip()
                    try:
                        np.savetxt(output_file, data.reshape(-1, data.shape[-1]) if data.ndim > 2 else data, 
                                  fmt='%.6f', delimiter='\t')
                        print(f"Saved to {output_file}")
                    except Exception as e:
                        print(f"Error saving: {e}")
                elif choice == "4":
                    break
                else:
                    print("Invalid choice")

if __name__ == "__main__":
    main()