#%% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

#%% Functions to implement different quantization schemes
def clamp(params, lb, ub):
    params_q = params
    params_q[params_q < lb] = lb
    params_q[params_q > ub] = ub
    return params_q

def asymmetric_quantization(params, bits):
    # Calculate the scale and zero point
    S = (np.max(params) - np.min(params)) / (2**bits-1)
    Z = -1*np.round(np.min(params) / S)
    Z = np.int32(Z)
    lb, ub = 0, 2**bits-1
    # Quantize the parameters
    params_q = clamp(np.round(params / S + Z), lb, ub).astype(np.int32)
    return params_q, S, Z

def symmetric_quantization(params, bits):
    # Calculate the scale
    S = np.max(np.abs(params)) / (2**(bits-1)-1)
    lb = -2**(bits-1)
    ub = 2**(bits-1)-1
    # Quantize the parameters
    params_q = clamp(np.round(params / S), lb, ub).astype(np.int32)
    return params_q, S

def quantization_error(params, params_q):
    # calculate the MSE
    return np.mean((params - params_q)**2)

def asymmetric_dequantize(params_q, S, Z):
    return (params_q - Z) * S

def symmetric_dequantize(params_q, S):
    return params_q * S



#%% Plotting the results
def plot_bullet_line(arr, label,axs):
    
    # Plotting the line
    axs.plot(arr, marker='o', linestyle='-')
    
    # Adding bullet points
    for i, val in enumerate(arr):
        axs.text(i+1, val, str(val), ha='center', va='bottom')
    
    axs.set_title(label)
    axs.legend()
    axs.grid(True)

def plot_error_bar(original_values, dequantized_values,label,axs):
    # Calculate error
    error = np.array(dequantized_values) - np.array(original_values)

    # Plotting the error bars
    axs.bar(range(len(original_values)), error, color='red', alpha=0.7)

    # Adding labels and title
    axs.set_xlabel('Index')
    axs.set_ylabel('Error')
    axs.set_title(label)
    axs.set_xticks(range(len(original_values)), original_values)

    axs.grid(axis='y')  # Add grid lines for better visualization of the error magnitudes

def plot_values_with_offset(values, offset, label, color):
    # Calculate the range for x-axis
    x_min = min(values)
    x_max = max(values)
    x_range = np.linspace(x_min, x_max, len(values))

    # Plotting the line with values as bullets and offset
    plt.scatter(x_range, np.full_like(x_range, offset), color='gray', marker='_', alpha=0.5)
    plt.scatter(values, np.full_like(values, offset), color=color, marker='o', label=label)

def plot_arrows(values1, values2, offset1, offset2, color):
    for v1, v2 in zip(values1, values2):
        plt.arrow(v1, offset1, v2 - v1, offset2 - offset1, color=color, alpha=0.3, width=0.01, head_width=0.0, head_length=0.0)


# Streamlit app
st.divider()
st.title("Quantization Demo \t")
st.divider()

# Sampling rate and quantization levels
lowNum = st.sidebar.number_input("Select the low value",value=-50)
highNum = st.sidebar.number_input("Select the high value",value=150)
div = st.sidebar.number_input("Select the Number of Division",value=20)
numBits = st.sidebar.slider("Select the Numberr of bits", min_value=1, max_value=64, value=8)

# Main Streamlit app
def main():   
    
    # Generate randomly distributed parameters    
    params = np.random.uniform(low=lowNum, high=highNum, size=div)
    
    #Sort the numbers for better display
    params = np.sort(params)
    
    # Round each number to the second decimal place
    params = np.round(params, 2)
    
    # Quantize to numBits
    (asymmetric_q, aS, aZ) = asymmetric_quantization(params, numBits)
    (symmetric_q, sS) = symmetric_quantization(params, numBits)
    
    # Display the DataFrame as a table
    # Create a dictionary with column names and data
    data = {
        'Original': params,
        'Asymmetric scale': asymmetric_q,
        'Symmetric scale': symmetric_q
        }
 
    df = pd.DataFrame(data)    
    st.table(df)
    
    # Dequantize the parameters back to 32 bits
    params_deq_asymmetric = asymmetric_dequantize(asymmetric_q, aS, aZ)
    params_deq_asymmetric = np.round(params_deq_asymmetric,2)
    params_deq_symmetric = symmetric_dequantize(symmetric_q, sS)
    params_deq_symmetric = np.round(params_deq_symmetric,2)
    
    # Display the DataFrame as a table
    # Create a dictionary with column names and data
    data = {
        'Original': params,
        'Dequantize Asymmetric': params_deq_asymmetric,
        'Dequantize Symmetric': params_deq_symmetric
        }
 
    df = pd.DataFrame(data)    
    st.table(df)
    
     
    # Calculate the quantization error
    st.write(f'{"Asymmetric error: ":>20}{np.round(quantization_error(params, params_deq_asymmetric), 2)}')
    st.write(f'{"Symmetric error: ":>20}{np.round(quantization_error(params, params_deq_symmetric), 2)}')
    
    # Display the results
    st.subheader("Original, Asymmetric and Symmetric Quantization") 
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    plot_bullet_line(params, "Original",axs[0])

    # Asymmetric Quantization
    plot_bullet_line(params_deq_asymmetric, "Asymmetric Quantization",axs[1])

    # Symmetric quantization
    plot_bullet_line(params_deq_symmetric, "Symmetric Quantization",axs[2])
    st.pyplot(fig)
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 3))
    plot_error_bar(params, params_deq_symmetric,'Error Symmetric Quantization',axs)
    st.pyplot(fig)
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 3))
    plot_error_bar(params, params_deq_asymmetric,'Error Asymmetric Quantization',axs)
    st.pyplot(fig)
    
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 3))

    # Plotting original values
    plot_values_with_offset(params, 0, 'Original', color='blue')
    
    # Plotting Dequantize Asymmetric values with offset
    plot_values_with_offset(params_deq_asymmetric, -0.2, 'Dequantize Asymmetric', color='red')
    
    # Plotting Dequantize Symmetric values with offset
    plot_values_with_offset(params_deq_symmetric, 0.2, 'Dequantize Symmetric', color='green')
    
    # Plotting arrows from top to below plots
    plot_arrows(params, params_deq_asymmetric, 0, -0.2, color='orange')
    plot_arrows(params, params_deq_symmetric, 0, 0.2, color='purple')
    
    # Adding labels and title
    plt.xlabel('Value')
    plt.title('Values Displayed with Offset')
    plt.legend()
    
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)  # Enable microgrid
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Disable y-axis
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()


