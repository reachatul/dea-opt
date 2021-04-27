# import click
import pandas as pd
import pulp as plp
import numpy as np
import itertools
import pickle

def build_model(stock_model, input, output, stocks):

    input = input.set_index("stock")
    output = output.set_index("stock")

    # Create a variable for weight of each input and output
    u = {col: plp.LpVariable(col, lowBound=0) for col in output.columns}

    v = {col: plp.LpVariable(col, lowBound=0) for col in input.columns}


    model = plp.LpProblem(stock_model, plp.LpMaximize)


    model += plp.lpSum(u[col]*output.at[stock_model, col] for col in output.columns)

    model += plp.lpSum(v[col]*input.at[stock_model, col] for col in input.columns) == 1

    for stock in stocks["stock_id"]:
        model += (plp.lpSum(u[col]*output.at[stock, col]
                            for col in output.columns) <=
                            plp.lpSum(v[col]*input.at[stock, col]
                                        for col in input.columns))

    status = model.solve(plp.PULP_CBC_CMD(msg=0))

    # print(f"Model {stock_model} is {plp.LpStatus[status]}")
    # model.writeLP(f"{stock_model}.lp")

    if plp.LpStatus[status] == 'Optimal':
        v_ = plp.value(plp.lpSum(v[col]*input.at[stock_model, col] for col in input.columns))
        u_ = plp.value(plp.lpSum(u[col]*output.at[stock_model, col] for col in output.columns))
        # print(f" Efficient for {stock_model} is {u_/v_}")
        output_efficiency[stock_model] = u_/v_
    else:
        output_efficiency[stock_model] = "inf"


def build_range_directional_model(stock_model, input, output, stocks):

    input = input.set_index("stock")
    output = output.set_index("stock")

    # create Range
    ro = output.max(axis=0) - output.loc[stock_model, :]
    ri = input.loc[stock_model, :] - input.min(axis=0)

    # Create a non negative Beta variable:
    beta = plp.LpVariable('beta', lowBound=0)

    # Create a lambda variable for each stock
    lmda = {stock: plp.LpVariable(f'l_{stock}', lowBound=0) for stock in
                                  stocks["stock_id"]}

    # maximize beta
    model = plp.LpProblem(stock_model, plp.LpMaximize); model+= beta

    model+= plp.lpSum(lmda[s] for s in stocks["stock_id"]) == 1

    for r in output.columns:
        model += (plp.lpSum(plp.lpSum(lmda[j]*output.at[j, r] for j in
        stocks['stock_id'])) >= output.at[stock_model, r] + beta*ro[r])

    for i in input.columns:
        model += (plp.lpSum(plp.lpSum(lmda[j]*input.at[j, i] for j in
        stocks['stock_id'])) <= input.at[stock_model, i] - beta*ri[i])

    status = model.solve(plp.PULP_CBC_CMD(msg=0))

    if plp.LpStatus[status] == 'Optimal':
        return 1 - plp.value(beta)
    else:
        'failed'


def dea(file="input"):
    xl = pd.ExcelFile(f"{file}.xlsx")
    xl.sheet_names
    stocks = xl.parse('stock')
    input = xl.parse('input')
    output = xl.parse('output')
    for i in range(1, len(input.columns)-1):
        input[f'input_{i}'] = pd.to_numeric(input[f'input_{i}'])

    for stock_model in stocks["stock_id"]:
        build_model(stock_model, input, output, stocks)
    efficiency = pd.DataFrame.from_dict(output_efficiency, orient='index',
                                        columns=["efficiency"])
    efficiency.to_csv(f"{file}_efficiency.csv")


def run_for_comb(numb=0):
    output_all = pickle.load(open("output_all.pkl", 'rb'))
    input_all = pickle.load(open("input_all.pkl", 'rb'))
    total_combs = pickle.load(open("total_combs.pkl", 'rb'))
    ror = pickle.load(open("ror.pkl", 'rb'))
    stocks = pickle.load(open("stocks.pkl", 'rb'))

    output_efficiency = {}
    input_cols_idx = np.where( total_combs[numb][0] == 1)
    input_cols = [f"input_{i+1}" for i in input_cols_idx[0]]

    output_cols_idx = np.where( total_combs[numb][1] == 1)
    output_cols = [f"output_{i+1}" for i in output_cols_idx[0]]


    input = input_all[["stock"] + input_cols]
    output = output_all[["stock"] + output_cols]

    for stock_model in stocks["stock_id"]:
        output_efficiency[stock_model] = build_range_directional_model(stock_model, input, output, stocks)
    efficiency = pd.DataFrame.from_dict(output_efficiency, orient='index',
                                        columns=["efficiency"])
    re = ror.merge(efficiency.reset_index().rename(columns={'index': 'stock'}), on = 'stock')
    return {numb: re['efficiency'].corr(re['rate_of_return'])}


def _get_effs(numb=0):
    output_all = pickle.load(open("output_all.pkl", 'rb'))
    input_all = pickle.load(open("input_all.pkl", 'rb'))
    total_combs = pickle.load(open("total_combs.pkl", 'rb'))
    ror = pickle.load(open("ror.pkl", 'rb'))
    stocks = pickle.load(open("stocks.pkl", 'rb'))

    output_efficiency = {}
    input_cols_idx = np.where( total_combs[numb][0] == 1)
    input_cols = [f"input_{i+1}" for i in input_cols_idx[0]]

    output_cols_idx = np.where( total_combs[numb][1] == 1)
    output_cols = [f"output_{i+1}" for i in output_cols_idx[0]]


    input = input_all[["stock"] + input_cols]
    output = output_all[["stock"] + output_cols]

    for stock_model in stocks["stock_id"]:
        output_efficiency[stock_model] = build_range_directional_model(stock_model, input, output, stocks)
    efficiency = pd.DataFrame.from_dict(output_efficiency, orient='index',
                                        columns=["efficiency"])
    return efficiency

if __name__ == '__main__':

    import multiprocessing
    pool = multiprocessing.Pool()
    result = pool.map(run_for_comb, range(130000))

    f = open('sol.pkl', 'wb')
    pickle.dump(result, f)
