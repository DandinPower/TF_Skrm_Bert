import matplotlib.pyplot as plt
import math
import numpy as np

def main():
    pm_count = [83894467488, 27179838720 ,3991538796 ,3848759936]
    app_count = [68515500240, 33219636480, 0, 0]
    width=1
    x_data = [10, 20, 30, 40]
    x2_data = [p + width for p in x_data]  
    x3_data = [p + 2*width for p in x_data]
    plt.bar(x2_data, pm_count, label = 'PermutationWrite')
    plt.bar(x3_data, app_count, label = 'Approximative')
    plt.xticks([p + width for p in x_data], x_data) 
    plt.legend()                                        
    plt.title('Algorithm Comparison')             
    plt.xlabel('Shift, Detect, Remove, Inject')                            
    plt.ylabel('Numbers Of Operations')  
    plt.show()

if __name__ == "__main__":
    main()
    #TestOne(-1.27, -1.28)