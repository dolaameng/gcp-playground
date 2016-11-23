from flask import Flask, render_template, request, Response
import json
import re

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.charts import TimeSeries, Donut, Bar

import string
import numpy as np

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

import pandas as pd
df = pd.read_csv("static/ab_result.csv")
df = df.set_index(pd.DatetimeIndex(df["time"]), drop=False)
df = df.sort_index()

app = Flask(__name__)
app.config["debug"] = True

@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/dataset")
def dataset():
    return render_template("dataset.html", data_table=df.to_html())
    
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question").encode("ascii", "ignore");
    
    answer = analyze(question);
    
    resp = Response(response = json.dumps(answer),
            status=200, mimetype="application/json")
    return resp;
    
def analyze(question):
    try:
        command = parse(question)
        print("DEBUG command:", command)
        if command["type"] == "count":
            plot_df = df.groupby("group").size().reset_index()
            plot_df.columns = ["group", "sample_size"]
            p = Donut(plot_df, label=["group", "sample_size"], values="sample_size", text_font_size="8pt",
                    hover_text="sample_size")
            script, div = components(p)
            answer = {
                "text": "Here is what I count for each group. <br> It is good that you have enough sample to test!",
                "plot": div,
                "script": script
            }
        elif command["type"] == "time-span":
            gdf = df.groupby("group")
            firsts = gdf.time.first()
            lasts = gdf.time.last()
            plot_df = pd.concat([firsts, lasts], axis=1)
            plot_df.columns=["start", "end"]
            msg = ""
            for g, s, e in plot_df.to_records():
                msg += "group %s starts at %s ends at %s<br>" % (g, s, e)
            answer = {
                "text": msg,
                "plot": 'Say "trend over time" to see a time plot',
                "script": ""
            }
        elif command["type"] == "time-plot":
            plot_df = df.groupby("group").apply(lambda s: s.resample("1M").success.mean()).T.reset_index()
            p = TimeSeries(plot_df.to_dict(orient="list"), x="time", y=list(plot_df.columns)[1:], legend=True,
                    title="monthly success rate", ylabel="success rate")
            script, div = components(p)
            answer = {
                "text": "got it",
                "plot": div,
                "script": script
            }
        elif command["type"] == "group-plot":
            fr, to = command["from"], command["to"]
            plot_df = df[fr:to]
            plot_df["success"] = plot_df["success"].astype(np.float)
            p = Bar(plot_df, "group", values="success", agg="mean",
                title="Success Rates by Group", color="wheat")
            script, div = components(p)
            msg = "Here is a comparison of group-wise success rate"
            if fr and to:
                msg += " from %s to %s" % (fr, to)
            answer = {
                "text": msg,
                "plot": div,
                "script": script
            }
        elif command["type"] == "likelihood":
            grps = df.group.unique()
            rows = []
            significance = []
            for ia in range(len(grps)):
                for ib in range(ia+1, len(grps)):
                    a, b = grps[ia], grps[ib]
                    ratea = df[df.group==a].success.mean()
                    rateb = df[df.group==b].success.mean()
                    sizea = df[df.group==a].shape[0]
                    sizeb = df[df.group==b].shape[0]
                    successa = df[df.group==a].success.sum()
                    successb = df[df.group==b].success.sum()
                    pool = (successa + successb) * 1. / (sizea + sizeb)
                    std = np.sqrt( pool * (1-pool) * (1./sizea + 1./sizeb) )
                    lb = (ratea-rateb) - 1.96 * std
                    ub = (ratea-rateb) + 1.96 * std
                    index = "%s vs %s" % (a, b)
                    if lb * ub > 0:
                        significance.append('"'+index+'"')
                    r = {
                        "index": index,
                        "rate diff": ratea-rateb,
                        "lower bound": lb,
                        "upper bound": ub
                    }
                    rows.append(r)
            plot_df = pd.DataFrame(rows).set_index("index")
            answer = {
                "text": "I think difference among %s is significant! Here is statistics:" % " and ".join(significance),
                "plot": plot_df.to_html(),
                "script": ""
            }
        else:
            answer = {
                "text": "Sorry I don't understand what you asked, I am still learning! :)",
                "plot": "",
                "script": ""
            }
    except Exception as e:
        print "ERROR HERE"
        raise e
        answer = {
                "text": "Sorry I don't understand what you asked, I am still learning! :)",
                "plot": "",
                "script": ""
        }
    return answer
    
def parse(question):
    print type(question), question
    q = question.lower().translate(None, string.punctuation)
    command = {}
    if re.search(r"(how.+many)|count", q) != None:
        command["type"] = "count"
    elif re.search(r"range|span", q) != None:
        command["type"] = "time-span"
    elif re.search(r"trend|time", q) != None:
        command["type"] = "time-plot"
    elif re.search(r"success|rate", q) != None and re.search(r"group|groups", q) != None:
        command["type"] = "group-plot"
        command["from"] = None
        command["to"] = None
        if re.search(r"from.+to", q) != None:
            fr = re.findall(r"from\s+(.+)\s+to", q)
            to = re.findall(r"to\s+(.+)$", q)
            command["from"] = fr[0] if fr else None
            command["to"] = to[0] if to else None
    elif re.search(r"compare|compares|better|worse", q) != None:
        command["type"] = "likelihood"
    else:
        command["type"] = "unknown"
    return command

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)