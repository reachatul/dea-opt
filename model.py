


import click
import pandas as pd



import pulp as plp

output_efficiency = {}

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

    print(f"Model {stock_model} is {plp.LpStatus[status]}")

    if plp.LpStatus[status] == 'Optimal':
        v_ = plp.value(plp.lpSum(v[col]*input.at[stock_model, col] for col in input.columns))
        u_ = plp.value(plp.lpSum(u[col]*output.at[stock_model, col] for col in output.columns))
        print(f" Efficient for {stock_model} is {u_/v_}")
        output_efficiency[stock_model] = u_/v_
    else:
        output_efficiency[stock_model] = "inf"


def dea(file="input"):
    xl = pd.ExcelFile(f"{file}.xlsx")
    xl.sheet_names
    stocks = xl.parse('stock')
    input = xl.parse('input')
    output = xl.parse('output')

    for stock_model in stocks["stock_id"]:
        build_model(stock_model, input, output, stocks)
    efficiency = pd.DataFrame.from_dict(output_efficiency, orient='index',
                                        columns=["efficiency"])
    efficiency.to_csv(f"{file}_efficiency.csv")

if __name__ == '__main__':
    dea("input")
    dea(file="report_20210120-01022")
