# import click
import pandas as pd
import pulp as plp

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
    # numb = 7374
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
    return efficiency.reset_index().rename(columns={'index': 'stock'})

if __name__ == '__main__':
    # read input and get all the combinations of input and output
    xl = pd.ExcelFile("input_data_IT_v2.xlsx")
    xl.sheet_names

    stocks = xl.parse('stock')
    input_all = xl.parse('input')
    output_all = xl.parse('output')
    ror = xl.parse('rate_of_return')

    import numpy as np
    import itertools

    input_num = input_all.shape[1] - 1
    output_num = output_all.shape[1] - 1

    input_combinations = np.array([list(j) for i, j in enumerate(
                itertools.product(*[[0, 1] for i in range(input_num)]))])
    input_combinations = [i for i in input_combinations if sum(i) >= 1]

    output_combinations = np.array([list(j) for i, j in enumerate(
                itertools.product(*[[0, 1] for i in range(output_num)]))])
    output_combinations = [i for i in output_combinations if sum(i) >= 1]

    total_combs = [(i, j) for i in input_combinations for j in output_combinations]

    len(total_combs)
    total_combs[0][0]
    total_combs[0][1]

    corr_comb = {}
    failed_ = []
    for i in range(len(total_combs)):
        # try: i = 7374
            efficiency = run_for_comb(i)
            re = ror.merge(efficiency, on = 'stock')
            corr_comb[i] = re['efficiency'].corr(re['rate_of_return'])
            if i%1000 == 0:
                print(f'completed {i}')
