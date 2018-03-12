from sklearn import linear_model
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score



def parse_data(data):
    all_data = json.loads(data)
    result = {}
    for vote in all_data:
        ori_data = vote["data"]
        if len(ori_data) == 0:
            #print("vote id ", vote["vote_id"], " has no action.")
            continue
        vote_data = {}
        ir = [item[0]["name"] for item in vote["initial_ranking"]]
        fr = [item[0]["name"] for item in vote["submitted_ranking"]]
        if len(ir) != len(fr):
            print("vote id ", vote["vote_id"], " initial and final ranking are of different length.")
            continue
        vote_data["ir"] = ir
        vote_data["fr"] = fr
        vote_data["time_submitted"] = float(vote["time_submission"]) / 1000
        if int(ori_data[0]["time"][0]) < 100000:
            vote_data["time_first_to_last"] = (float(ori_data[len(ori_data)-1]["time"][1]) - float(ori_data[0]["time"][0])) / 1000
        else:
            vote_data["time_first_to_last"] = (float(ori_data[len(ori_data)-1]["time"][1]) - float(ori_data[1]["time"][0])) / 1000
        vote_data["kt"] = KTDistance(ir,fr)
        vote_data["n_kt"] = NKTDistance(vote_data["kt"],ir,fr)
        vote_data["misplacement"] = misplacement(ir,fr)
        user = vote["user_id"]
        if user in result.keys():
            result[user].append(vote_data)
        else:
            result[user] = []
            result[user].append(vote_data)
    print("All data parsed.")
    #print(result[1387])
    #record_length = [len(record) for record in result.values()]
    #long_record = [r for r in record_length if r >= 20]
    #print (len(long_record))
    return result
    
def translate_data_for_regression(data):
    result = {}
    for user,records in data.items():
        if len(records) < 20:
            continue
        y1 = [] #submit_time
        y2 = [] #time between first action and last action
        x1 = [] #KT Distance
        x2 = [] #Normalized KT Distance
        x3 = [] #Misplacement
        for r in records:
            y1.append(r["time_submitted"])
            y2.append(r["time_first_to_last"])
            x1.append(r["kt"])
            x2.append(r["n_kt"])
            x3.append(r["misplacement"])
        train_data = [y1,y2,x1,x2,x3]
        result[user] = train_data
    return result
    
    
def learning_by_user(type,train_data):
    train_x = np.reshape(train_data[type][:-5],(-1,1))
    test_x = np.reshape(train_data[type][-5:],(-1,1))
    train_y = train_data[1][:-5]
    test_y = train_data[1][-5:]
    
    regr = linear_model.LinearRegression()
    regr.fit(train_x,train_y)
    
    pred_y = regr.predict(test_x)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f"
    #      % mean_squared_error(test_y, pred_y))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(test_y, pred_y))
    return regr.coef_, mean_squared_error(test_y, pred_y),r2_score(test_y, pred_y), explained_variance_score(test_y, pred_y)
    # Plot outputs
    #plt.scatter(test_x, test_y,  color='black')
    #plt.plot(test_x, pred_y, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()
    
    
def learning_by_user_with_two_features(train_data):
    train_x = np.reshape([train_data[2][:-5],train_data[4][:-5]],(-1,2))
    test_x = np.reshape([train_data[2][-5:],train_data[4][-5:]],(-1,2))
    train_y = train_data[1][:-5]
    test_y = train_data[1][-5:]
    
    regr = linear_model.LinearRegression()
    regr.fit(train_x,train_y)
    
    pred_y = regr.predict(test_x)
    
    return regr.coef_, mean_squared_error(test_y, pred_y),r2_score(test_y, pred_y), explained_variance_score(test_y, pred_y)

        
def KTDistance(rank1, rank2):
    pairwise_diff = 0
    for i in range(len(rank1)):
        for j in range(i+1,len(rank1)):
            if rank2.index(rank1[i]) > rank2.index(rank1[j]):
                pairwise_diff += 1
    return pairwise_diff
    
def NKTDistance(KT,rank1,rank2):
    return KT*2 / (len(rank1)*(len(rank2)-1))
    

def misplacement(rank1,rank2):
    opt = np.zeros((len(rank1)+1,len(rank2)+1),dtype='int32')
    for i in range(1,len(rank1)+1):
        for j in range(1,len(rank2)+1):
            if rank1[i-1] == rank2[j-1]:
                opt[i,j] = opt[i-1,j-1] + 1
            else:
                opt[i,j] = max(opt[i-1,j],opt[i,j-1])
                
    return len(rank1) - opt[len(rank1),len(rank2)]
    
if __name__ == "__main__":
    file = open('RankNumber.json','r')
    data = file.read()
    file.close()
    parsed_data = parse_data(data)
    train_data = translate_data_for_regression(parsed_data)
    #print(train_data)
    data_size = len(train_data.keys())
    altered_data_size = data_size
    #learning_by_user_single(4,train_data[983])
    kt_best = 0
    m_best = 0
    c_best = 0
    kt_best_out = 0
    m_best_out = 0
    c_best_out = 0
    #print(train_data[726])
    total_mean_error = 0
    for user,d in train_data.items():
        KT_result = learning_by_user(2,d)
        M_result = learning_by_user(4,d)
        C_result = learning_by_user_with_two_features(d)
        
        best_mean_square = min(KT_result[1],M_result[1],C_result[1])
        if best_mean_square == KT_result[1]:
            kt_best += 1
        elif best_mean_square == M_result[1]:
            m_best += 1
        else:
            c_best += 1
        if best_mean_square < 300:
            total_mean_error += best_mean_square
            if best_mean_square == KT_result[1]:
                kt_best_out += 1
            elif best_mean_square == M_result[1]:
                m_best_out += 1
            else:
                c_best_out += 1
        else:
            altered_data_size -= 1
        #print("User ",user,"'s result:\nKT--- Coefficients: ", KT_result[0],"Mean squared error: %.2f"%KT_result[1]," r2 Variance score: %.2f"%KT_result[2],"explained variance score: %.2f"%KT_result[3])
        #print("Misplacement--- Coefficients: ", M_result[0],"Mean squared error: %.2f"%M_result[1]," r2 Variance score: %.2f"%M_result[2],"explained variance score: %.2f"%M_result[3])
        #print("Combined--- Coefficients: ", C_result[0],"Mean squared error: %.2f"%C_result[1]," r2 Variance score: %.2f"%C_result[2],"explained variance score: %.2f"%M_result[3])
    average_mean_error = total_mean_error / altered_data_size
    print("Total data amount:", data_size, ", out of which",kt_best,"instances KT is best,",m_best, "instances Misplacement is best,", c_best, "instances combined is best")
    print("Data amount after eliminating outliers:", altered_data_size, ", out of which",kt_best_out,"instances KT is best,",m_best_out, "instances Misplacement is best,", c_best_out, "instances combined is best")
    print("Average square error: %.2f"%average_mean_error, "after eliminating outliers,",altered_data_size,"users are counted")