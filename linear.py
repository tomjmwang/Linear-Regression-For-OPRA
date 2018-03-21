from sklearn import linear_model
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import copy
import pickle


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
            vote_data["number_of_moves"] = len(ori_data)
            vote_data["time_of_first_move"] = float(ori_data[0]["time"][0]) / 1000
            vote_data["ave_each_action_time"] = sum([float(d["time"][1])-float(d["time"][0]) for d in ori_data ])/len(ori_data) / 1000
            total_interval = 0.0
            for i in range(1,len(ori_data)):
                total_interval += float(ori_data[i]["time"][0]) - float(ori_data[i-1]["time"][1])
            if len(ori_data) == 1:
                vote_data["ave_interval_between_actions"] = 0
            else:
                vote_data["ave_interval_between_actions"] = total_interval / (len(ori_data)-1) / 1000
        else:
            vote_data["time_first_to_last"] = (float(ori_data[len(ori_data)-1]["time"][1]) - float(ori_data[1]["time"][0])) / 1000
            vote_data["number_of_moves"] = len(ori_data) - 1
            vote_data["time_of_first_move"] = float(ori_data[1]["time"][0]) / 1000
            vote_data["ave_each_action_time"] = sum([float(d["time"][1])-float(d["time"][0]) for d in ori_data[1:] ])/(len(ori_data)-1) / 1000
            total_interval = 0.0
            for i in range(2,len(ori_data)):
                total_interval += float(ori_data[i]["time"][0]) - float(ori_data[i-1]["time"][1])
            vote_data["ave_interval_between_actions"] = total_interval / (len(ori_data)-2) / 1000
        vote_data["kt"] = KTDistance(ir,fr)
        vote_data["n_kt"] = NKTDistance(vote_data["kt"],ir,fr)
        vote_data["misplacement"] = misplacement(ir,fr)
        vote_data["action_sequence"] = calculate_action_sequence(ori_data,ir,fr)
        for i in range(10):
            name = "f" + str(i)
            vote_data[name] = [action_similarity(vote_data["action_sequence"],inserlection_predictor(i,copy.deepcopy(ir),copy.deepcopy(fr))[1])]
            vote_data[name].append(action_distance(vote_data["action_sequence"],inserlection_predictor(i,copy.deepcopy(ir),copy.deepcopy(fr))[1]))
            #print(name,vote_data[name])
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

#input: action data (list of dictionaries), initial ranking (list of string), final ranking (list of string)
#output: list of tuples of three items
def calculate_action_sequence(data,ir,fr):
    seq = []
    if int(data[0]["time"][0]) > 100000:
        data = data[1:]
    for action in data:
        #print(action)
        try:
            first_rank = [item[0]["name"] for item in action["rank"][0]]
            last_rank = [item[0]["name"] for item in action["rank"][1]]
            #print(first_rank,last_rank)
            item = action["item"]
            vector = (fr.index(item),first_rank.index(item),last_rank.index(item))
        except:
            vector = (0,0,0)
        seq.append(vector)
    return seq

def action_similarity(action1,action2):
    opt = np.zeros((len(action1)+1,len(action2)+1),dtype='int32')
    #print(action1,action2)
    for i in range(1,len(action1)+1):
        for j in range(1,len(action2)+1):
            if action1[i-1] == action2[j-1]:
                opt[i,j] = opt[i-1,j-1] + 1
            else:
                opt[i,j] = max(opt[i-1,j],opt[i,j-1])
    return opt[len(action1),len(action2)]

def single_penalty(vector):
    return vector[0]**2 + vector[1]**2 + vector[2]**2

def aligned_distance(vector1,vector2):
    return (vector1[0] - vector2[0])**2 + (vector1[1] - vector2[1])**2 + (vector1[2] - vector2[2])**2

#action1 -- user's action
#action2 -- expected action
def action_distance(action1, action2):
    opt = np.zeros((len(action1)+1,len(action2)+1),dtype='int32')
    #print(action1,action2)
    for i in range(1,len(action1)+1):
        opt[i,0] = opt[i-1,0] + single_penalty(action1[i-1])
    for i in range(1,len(action2)+1):
        opt[0,i] = opt[0,i-1] + single_penalty(action2[i-1])
    for i in range(1,len(action1)+1):
        for j in range(1,len(action2)+1):
            opt[i,j] = min(opt[i-1,j]+single_penalty(action1[i-1]),opt[i,j-1]+single_penalty(action2[j-1]),opt[i-1,j-1]+aligned_distance(action1[i-1],action2[j-1]))
    return opt[len(action1),len(action2)]
    
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
        x4 = [] #time between page loading and first action
        x5 = [] #number of moves
        x6 = [] #average time for each action for each poll
        x7 = [] #average time between each action for each poll
        total_similarity = {}
        total_distance = {}
        for i in range(10):
            name = "f" + str(i)
            total_similarity[name] = 0
            total_distance[name] = 0
        for r in records:
            y1.append(r["time_submitted"])
            y2.append(r["time_first_to_last"])
            x1.append(r["kt"])
            x2.append(r["n_kt"])
            x3.append(r["misplacement"])
            x4.append(r["time_of_first_move"])
            x5.append(r["number_of_moves"])
            x6.append(r["ave_each_action_time"])
            x7.append(r["ave_interval_between_actions"])
            for i in range(10):
                name = "f" + str(i)
                total_similarity[name] += r[name][0]
                total_distance[name] += r[name][1]
        train_data = [y1,y2,x1,x2,x3,x4,x5,x6,x7,total_similarity,total_distance]
        result[user] = train_data

    return result

def cluster_users(data):
    clusters_sim = [{},{},{},{},{},{},{},{},{},{}]
    clusters_dist = [{},{},{},{},{},{},{},{},{},{}]
    for user,d in data.items():
        sim = 0
        dist = 99999
        final_f_sim = "f0"
        final_f_dist = "f0"
        for f,value in d[9].items():
            #print(value)
            if value > sim:
                sim = value
                final_f_sim = f
        for f,value in d[10].items():
            if value < dist:
                dist = value
                final_f_dist = f

        ind = int(final_f_sim[1:])
        clusters_sim[ind][user] = d[:-2]
        ind = int(final_f_dist[1:])
        clusters_dist[ind][user] = d[:-2]
    lengths_sim = [len(c) for c in clusters_sim]
    lengths_dist = [len(c) for c in clusters_dist]
    print(lengths_sim)
    print(lengths_dist)
    return clusters_sim, clusters_dist
    
    
def learning_by_user(type,train_data):
    #m_x = max(train_x)
    #for n in range(len(train_x)):
    #   train_x[n] = train[n]/m_x
    train_x = np.reshape(train_data[type],(-1,1))
    #test_x = np.reshape(train_data[type][-5:],(-1,1))
    train_y = train_data[0]
    #test_y = train_data[1][-5:]
    
    regr = linear_model.LinearRegression()
    regr.fit(train_x,train_y)
    
    #pred_y = regr.predict(test_x)
    pred_y = regr.predict(train_x)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f"
    #      % mean_squared_error(test_y, pred_y))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(test_y, pred_y))
    return regr.coef_, mean_squared_error(train_y, pred_y),r2_score(train_y, pred_y), explained_variance_score(train_y, pred_y),regr.intercept_
    # Plot outputs
    #plt.scatter(test_x, test_y,  color='black')
    #plt.plot(test_x, pred_y, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()
    
    
def learning_by_user_with_two_features(train_data):
    #m_x_1 = max(train_data[2])
    #train_x_1 = train_data[2]
    #for n in range(len(train_x_1))
        #train_x_1[]
    #train_x_2 = train_data[4]
    train_x = np.reshape([train_data[2],train_data[4]],(-1,2))
    #test_x = np.reshape([train_data[2][-5:],train_data[4][-5:]],(-1,2))
    train_y = train_data[0]
    #test_y = train_data[1][-5:]
    
    regr = linear_model.LinearRegression()
    regr.fit(train_x,train_y)
    
    pred_y = regr.predict(train_x)
    
    return regr.coef_, mean_squared_error(train_y, pred_y),r2_score(train_y, pred_y), explained_variance_score(train_y, pred_y),regr.intercept_
    
def learning_by_user_with_multiple_features(feature_list,train_data):
    x_list = []
    for f in feature_list:
        x_list.append(train_data[f])
    #print(x_list)
    for l in range(len(x_list)):
        max_x = max(x_list[l])
        for j in range(len(x_list[l])):
            x_list[l][j] = x_list[l][j]/max_x
    #print(x_list)
    if len(x_list) == 1:
        train_x = np.reshape(x_list[0],(-1,len(feature_list)))
    else:
        train_x = np.reshape(x_list,(-1,len(feature_list)))
    train_y = train_data[0]
    regr = linear_model.LinearRegression()
    regr.fit(train_x,train_y)
    pred_y = regr.predict(train_x)
    return regr.coef_, mean_squared_error(train_y, pred_y),r2_score(train_y, pred_y), explained_variance_score(train_y, pred_y),regr.intercept_


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

def selection_predictor(rank1,rank2):
    moves = []
    dist = 0
    for i in range(len(rank2)):
        if rank1[i] != rank2[i]:
            for j in range(i+1,len(rank1)):
                if rank1[j] == rank2[i]:
                    rank1.insert(i,rank1.pop(j))
                    action = (i,j,i)
                    moves.append(action)
                    dist += j-i
                    break
    return (len(moves),moves,dist)

def insertion_predictor(rank1,rank2):
    moves = []
    dist = 0
    for i in range(1,len(rank1)):
        for j in range(0,i):
            if rank2.index(rank1[j]) > rank2.index(rank1[i]):
                action = (rank2.index(rank1[i]),i,j)
                rank1.insert(j,rank1.pop(i))
                moves.append(action)
                dist += i-j
                break
    return (len(moves),moves,dist)

def inserlection_predictor(num,rank1,rank2):
    if num == 0:
        return insertion_predictor(rank1,rank2)
    if num >= len(rank1)-1:
        return selection_predictor(rank1,rank2)
    moves = []
    dist = 0
    for i in range(num):
        if rank1[i] != rank2[i]:
            for j in range(i+1,len(rank1)):
                if rank1[j] == rank2[i]:
                    rank1.insert(i,rank1.pop(j))
                    action = (i,j,i)
                    moves.append(action)
                    dist += j-i
                    break
    for i in range(num+1,len(rank1)):
        for j in range(num,i):
            if rank2.index(rank1[j]) > rank2.index(rank1[i]):
                dist += i-j
                action = (rank2.index(rank1[i]),i,j)
                rank1.insert(j,rank1.pop(i))
                moves.append(action)
                break
    #print(rank1)
    return (len(moves),moves,dist)

def learning_action(train_data,feature_list):
    data_size = len(train_data.keys())
    altered_data_size = data_size
    mse_list = []
    multiple_f_mse = 0
    for user,d in train_data.items():
        #if user == 268:
            #print(d)
        T_result = learning_by_user_with_multiple_features(feature_list,d)
        if T_result[1] < 100:
            mse_list.append(T_result[1])
            multiple_f_mse += T_result[1]
            #print(T_result)
        else:
            altered_data_size -= 1
        #print("User ",user,"'s result:\nKT--- Coefficients: ", T_result[0],T_result[4],"Mean squared error: %.2f"%T_result[1]," r2 Variance score: %.2f"%T_result[2],"explained variance score: %.2f"%T_result[3])
    return multiple_f_mse/altered_data_size

def feature_name(num):
    if num == 2:
        return "KT"
    elif num == 3:
        return "NKT"
    elif num == 4:
        return "Misp"
    elif num == 5:
        return "TBA"
    elif num == 6:
        return "NoM"
    elif num == 7:
        return "ToEA"
    else:
        return "TBEA"

    
if __name__ == "__main__":
    #for i in range(10):
        #print(inserlection_predictor(i,[2,4,1,6,3,5,0,9,8,7],[7,8,6,1,3,2,4,0,5,9]))
    
    #file = open('RankNumber.json','r')
    #data = file.read()
    #file.close()
    #parsed_data = parse_data(data)
    #train_data = translate_data_for_regression(parsed_data)
    #clustered_data = cluster_users(train_data)
    s_c_file = open("similarity_clusters","rb")
    d_c_file = open("distance_clusters","rb")
    clustered_data_similarity = pickle.load(s_c_file)
    clustered_data_distance = pickle.load(d_c_file)
    #pickle.dump(clustered_data[0],s_c_file)
    #pickle.dump(clustered_data[1],d_c_file)
    clustered_data = clustered_data_similarity,clustered_data_distance
    s_c_file.close()
    d_c_file.close()
    train_data = clustered_data[0][0]

    f_names = ["f0","f1","f2","f3","f4","f5","f6","f7","f8"]
    sim_len = [len(c) for c in clustered_data_distance[:-1]]

    plt.bar(f_names,sim_len)
    plt.show()

    """
    feature_list = [2,4,5,6,7,8]
    mse_x1 = []
    mse_x2 = []
    mse_x3 = []
    mse_x4 = []
    mse_x5 = []
    mse_x6 = []
    mse_all = []

    for c in clustered_data_distance[:-1]:
        mse_x1.append(learning_action(c,[2]))
        mse_x2.append(learning_action(c,[4]))
        mse_x3.append(learning_action(c,[5]))
        mse_x4.append(learning_action(c,[6]))
        mse_x5.append(learning_action(c,[7]))
        mse_x6.append(learning_action(c,[8]))
        mse_all.append(learning_action(c,feature_list))
    print(mse_x1)
    x = [4,12,20,28,36,44,52,60,68]
    ax = plt.subplot(111)
    ax.bar([n-3 for n in x],mse_x1,width=1,color='b',align='center')
    ax.bar([n-2 for n in x],mse_x2,width=1,color='g',align='center')
    ax.bar([n-1 for n in x],mse_x3,width=1,color='r',align='center')
    ax.bar([n for n in x],mse_x4,width=1,color='y',align='center')
    ax.bar([n+1 for n in x],mse_x5,width=1,color='orange',align='center')
    ax.bar([n+2 for n in x],mse_x6,width=1,color='purple',align='center')
    ax.bar([n+3 for n in x],mse_all,width=1,color='black',align='center')

    plt.show()
    

    while True:
        try:
            print("Enter command")
            command = input()
            if command == "cluster":
                print("Enter cluster type: (0 is similarity, 1 is distance)")
                cluster_type = int(input())
                print("Enter f number: (0-8)")
                cluster_number = int(input())
                learning_action(clustered_data[cluster_type][cluster_number])
            elif command == "all":
                pass
            elif command == "quit":
                break
            else:
                print("Unknown command")
        except:
            print("Some error occurred, try again")
            continue



    #print(train_data)
    data_size = len(train_data.keys())
    altered_data_size = data_size
    #learning_by_user_single(4,train_data[983])
    #best data before eliminating outliers
    kt_best = 0
    m_best = 0
    c_best = 0
    #best data after eliminating outliers
    kt_best_out = 0
    m_best_out = 0
    c_best_out = 0
    #print(train_data[726])
    total_mean_error = 0
    #total MSE for each method
    tmkt = 0
    tmm = 0
    tmc = 0
    #total submit time
    tt = 0
    #total submit time entries
    tt_e = 0
    #total first_last time
    tt2 = 0
    #total first_last time entries
    tt_e2 = 0

    #total se for regression with no feature (horizontal line)
    t_mse = 0
    for user,d in train_data.items():
        ave_t = sum(d[0])/len(d[0])
        mse = sum([(n-ave_t)*(n-ave_t) for n in d[0]])
        if mse < 200:
            t_mse += mse
        else:
            altered_data_size -= 1
    a_mse = t_mse / altered_data_size
    #print(ave_t, a_mse)
    
    altered_data_size = data_size
    
    mse_list = []
    multiple_f_mse = 0
    #print(train_data.keys())
    for user,d in train_data.items():
        #if user == 268:
            #print(d)
        feature_list = [4,5,6,7,8]
        T_result = learning_by_user_with_multiple_features(feature_list,d)
        if T_result[1] < 200:
            mse_list.append(T_result[1])
            multiple_f_mse += T_result[1]
            #print(T_result)
        else:
            altered_data_size -= 1
        #print("User ",user,"'s result:\nKT--- Coefficients: ", T_result[0],T_result[4],"Mean squared error: %.2f"%T_result[1]," r2 Variance score: %.2f"%T_result[2],"explained variance score: %.2f"%T_result[3])
    print(multiple_f_mse/altered_data_size, altered_data_size)
    b = []
    i = 0
    while i < 200:
        b.append(i)
        i+=5
    #plt.hist(mse_list,bins=b)
    #plt.show()

    altered_data_size = data_size
    best_kt_sum = 0
    best_m_sum = 0
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
        if best_mean_square < 200:
            total_mean_error += best_mean_square
            tmkt += KT_result[1]
            tmm += M_result[1]
            tmc += C_result[1]
            tt += sum(d[0])
            tt_e += len(d[0])
            tt2 += sum(d[1])
            tt_e2 += len(d[1])
            if best_mean_square == KT_result[1]:
                kt_best_out += 1
                best_kt_sum += best_mean_square
            elif best_mean_square == M_result[1]:
                m_best_out += 1
                best_m_sum += best_mean_square
            else:
                c_best_out += 1
        else:
            altered_data_size -= 1
        #if user == 275:
            #print("User ",user,"'s result:\nKT--- Coefficients: ", KT_result[0],KT_result[4],"Mean squared error: %.2f"%KT_result[1]," r2 Variance score: %.2f"%KT_result[2],"explained variance score: %.2f"%KT_result[3])
            #print("Misplacement--- Coefficients: ", M_result[0],M_result[4],"Mean squared error: %.2f"%M_result[1]," r2 Variance score: %.2f"%M_result[2],"explained variance score: %.2f"%M_result[3])
            #print("Combined--- Coefficients: ", C_result[0],C_result[4],"Mean squared error: %.2f"%C_result[1]," r2 Variance score: %.2f"%C_result[2],"explained variance score: %.2f"%M_result[3])
    average_mean_error = total_mean_error / altered_data_size
    amkt = tmkt / altered_data_size
    atmm = tmm / altered_data_size
    atmc = tmc / altered_data_size
    att = tt/tt_e
    att2 = tt2/tt_e2
    #print(best_kt_sum/kt_best_out,best_m_sum/m_best_out)
    #print("Total data amount:", data_size, ", out of which",kt_best,"instances KT is best,",m_best, "instances Misplacement is best,", c_best, "instances combined is best")
    #print("Data amount after eliminating outliers:", altered_data_size, ", out of which",kt_best_out,"instances KT is best,",m_best_out, "instances Misplacement is best,", c_best_out, "instances combined is best")
    #print("Data amount after eliminating outliers:", att,att2, ", out of which",amkt,"instances KT is best,",atmm, "instances Misplacement is best,", atmc, "instances combined is best")
    #print("Average square error: %.2f"%average_mean_error, "after eliminating outliers,",altered_data_size,"users are counted")
    #print(train_data[275])

    #plt.scatter(test_x, test_y,  color='black')
    """


    


    