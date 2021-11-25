import numpy as np
import pandas as pd
from util import *
df = pd.DataFrame(np.random.rand(150,len(fingerprinting_metrics)), columns=fingerprinting_metrics)

matrix = np.asmatrix(np.random.rand(len(recommendations),len(fingerprinting_metrics)))

def cosine_similarity(list1, list2):
        dotprod = 0
        betrag_x = 0
        betrag_y = 0
        for i in range(0,len(list1)):
            x=list1[i]
            y=list2[i]
            dotprod+=x*y
            betrag_x+=x*x
            betrag_y+=y*y
        return dotprod/((betrag_x*betrag_y)**(1/2.0))

def give_recommendation_based_on_correlation_matrix(matrix, fingerprinting_vector):
    cos_sim_list = [{"id":x,"sim":cosine_similarity(matrix[x],fingerprinting_vector)} for x in range(0, len(matrix))]
    cos_sim_list.sort(key = lambda x: x["sim"], reverse=True)
    recommendation = recommendations[cos_sim_list[0]["id"]]
    return recommendation
res = [give_recommendation_based_on_correlation_matrix(matrix.tolist(),row.to_list()) for i,row in df.iterrows()]
df["top 1 recommendation"] = res
