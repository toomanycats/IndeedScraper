from flask import Flask, render_template
from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.util.string import encode_utf8
from bokeh.plotting import figure
import pandas as pd

app = Flask(__name__)

@app.route('/')
def test():
    kws = ["one", "two", "cat", "dog"]
    count = [23, 45, 11, 87]
    df = pd.DataFrame({"kw": kws,
                       "count": count
                       })

    #p = Bar(df, 'kw')

    df.sort("count", inplace=True)
    df.set_index("kw", inplace=True)
    series = df['count']

    p = figure(width=1000, height=1000, y_range=series.index.tolist())

    j = 1
    for k, v in series.iteritems():
        w = v / 2 * 2
        p.rect(x=v/2,
               y=j,
               width=w,
               height=0.4,
               color=(76, 114, 176),
               width_units="data",
               height_units="data"
                )
        j += 1

    script, div = components(p)

    page = render_template('test.html', div=div, script=script)
    return encode_utf8(page)


if __name__ == "__main__":
    app.run(debug=True,
            threaded=False
            )
