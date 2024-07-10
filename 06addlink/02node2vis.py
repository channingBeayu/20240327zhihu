import pickle

with open(r'F:\Code\240313\gat\graph\dataset\mydata\data_nodes.pkl', "rb+") as f:
    data_nodes = pickle.load(f)
with open('data/data_edges.pkl', "rb+") as f:
    data_edges = pickle.load(f)


f = open('travel_event_graph.html', 'w+', encoding='utf-8')
base = '''
    <html>
    <head>
      <script type="text/javascript" src="util/VIS/dist/vis.js"></script>
      <link href="util/VIS/dist/vis.css" rel="stylesheet" type="text/css">
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    </head>
    <body>

    <div id="VIS_draw"></div>
    <script type="text/javascript">
      var nodes = data_nodes;
      var edges = data_edges;

      var container = document.getElementById("VIS_draw");

      var data = {
        nodes: nodes,
        edges: edges
      };

      var options = {
          nodes: {
              shape: 'dot',
              size: 25,
              font: {
                  size: 14
              }
          },
          edges: {
              font: {
                  size: 14,
                  align: 'middle'
              },
              color: 'gray',
              arrows: {
                  to: {enabled: true, scaleFactor: 0.5}
              },
              smooth: {enabled: false}
          },
          physics: {
              enabled: true
          }
      };

      var network = new vis.Network(container, data, options);

    </script>
    </body>
    </html>
    '''
html = base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
f.write(html)
f.close()