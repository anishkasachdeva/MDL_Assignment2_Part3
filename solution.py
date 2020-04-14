import os
import numpy as np
import cvxpy as cvx
import shutil
import json

reward=-20
total_actions=0
actions_arr=[]
reward_arr=[]
def get_action(i):
    if i==1:
        return "NOOP"
    if i==2:
        return "SHOOT"
    if i==3:
        return "DODGE"
    if i==4:
        return "RECHARGE"
def initialize_actions(actions,r):
    global reward
    global total_actions
    for i in range(60):
        a=[-1,-1,-1,-1]
        # b=[]
        stamina=(i%3)*50
        arrows=int((i%12)/3)
        health=int(i/12)*25
        if health==0:
            a[0]=0
            # b.append(0)
            actions.append(a)
            r.append(0)
            total_actions=total_actions+1
            continue
        if stamina>0 and arrows>0:
            a[1]=reward
            r.append(reward)
            total_actions=total_actions+1
        actions.append(a)
        if stamina>0:
            a[2]=reward
            r.append(reward)
            total_actions=total_actions+1
        if stamina<100:
            a[3]=reward
            r.append(reward)
            total_actions=total_actions+1
        # r.append(b)
def probablity(state,action):
    prob=[]
    st=[]
    stamina=(state%3)*50
    arrows=int((state%12)/3)
    # health=int(state/12)*25
    if action==2:
        prob.append(0.5)
        st.append(state-1-3)
        prob.append(0.5)
        st.append(state-1-3-12)
    if action==1:
        prob.append(1)
        st.append(state)
        return prob,st
    if action==3:
        if stamina==100:
            if arrows==3:
                prob.append(0.8)
                st.append(state-1)
                prob.append(0.2)
                st.append(state-2)
            else:
                prob.append(0.64)
                st.append(state-1+3)
                prob.append(0.04)
                st.append(state-2)
                prob.append(0.16)
                st.append(state-2+3)
                prob.append(0.16)
                st.append(state-1)
        else:
            if arrows==3:
                prob.append(1)
                st.append(state-1)
            else:
                prob.append(0.2)
                st.append(state-1)
                prob.append(0.8)
                st.append(state-1+3)
    if action==4:
        prob.append(0.8)
        st.append(state+1)
        prob.append(0.2)
        st.append(state)
    return prob,st

def make_policy(a,x):
    p=[]
    current_action_index=0
    for i in range(60):
        stamina=(i%3)
        arrows=int((i%12)/3)
        health=int(i/12)
        state=[]
        state.append(health)
        state.append(arrows)
        state.append(stamina)
        actions_in_this_state=[]
        x_for_these_actions=[]
        for j in range(4):
            if a[i][j]!=-1:
                actions_in_this_state.append(j+1)
                x_for_these_actions.append(x[current_action_index])
                current_action_index=current_action_index+1
        actions_in_this_state=np.array(actions_in_this_state)
        x_for_these_actions=np.array(x_for_these_actions)
        max_x_index=np.argmax(x_for_these_actions)
        temp=[]
        temp.append(state)
        # print(state)
        # print(actions_in_this_state)
        # print(x_for_these_actions)
        # print(max_x_index)
        temp.append(get_action(actions_in_this_state[max_x_index]))
        p.append(temp)
    # print(json.dumps(p, indent=True))
    return p


try:
    if os.path.exists('./outputs'):
        shutil.rmtree('./outputs')
    os.mkdir('./outputs')
except OSError as error:
    print(error)
    sys.exit()


# def produce_output():
#     with open("./outputs/output.json", "a") as file:
#         result = solver()
#         json.dump(result, file)


A=[]
initialize_actions(actions_arr,reward_arr)
for i in range(60):
    temp=[]
    for j in range(total_actions):
        temp.append(0)
    A.append(temp)
action_coloumn=0
for i in range(60):
    for j in range(4):
        if actions_arr[i][j]!=-1:
            # print(i,action_coloumn)
            prob_arr,state_arr=probablity(i,j+1)
            if int(i/12):
                A[i][action_coloumn]=1
            else:
                A[i][action_coloumn]=2
            for k in range(len(state_arr)):
                A[state_arr[k]][action_coloumn]=A[state_arr[k]][action_coloumn]-prob_arr[k]
            action_coloumn=action_coloumn+1



shape=(len(reward_arr),1)
X=cvx.Variable(shape)
A=np.array(A)
alpha=[]
for i in range(60):
    temp=[]
    temp.append(0)
    alpha.append(temp)
alpha[59][0]=1
alpha=np.array(alpha)
constraint=[cvx.matmul(A,X)==alpha,X>=0]
reward_arr=np.array(reward_arr)
objective=cvx.Maximize(cvx.sum(cvx.matmul(reward_arr,X)))
prob=cvx.Problem(objective,constraint)
solution=prob.solve(solver=cvx.CVXOPT)
X_value=X.value.flatten().tolist()


policy=make_policy(actions_arr,X_value)

# print(json.dumps(X_value, indent=True))


dictionary = {"a": A.tolist(),"r":reward_arr.tolist(), "alpha": alpha.flatten().tolist(), "x": X_value, "policy": policy, "objective": solution}
# print(solution)
# print(X.value)
file=open("./outputs/output.json", "a")
json.dump(dictionary,file)
file.close()