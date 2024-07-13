import random
from py2neo import Graph, NodeMatcher

graph = Graph('bolt://localhost:7687', auth=('neo4j', '123456'))

###########无果


def find_path(graph, start, end):
    # paths = graph.run("MATCH p=shortestPath((start)-[*]-(end)) RETURN p", start=start["self"], end=end["self"]).data()
    paths = graph.run("MATCH p=shortestPath((start)-[*]-(end)) RETURN p", start=start, end=end).data()
    return [list(path["p"].nodes()) for path in paths]

####
node_matcher = NodeMatcher(graph)
start = node_matcher.match(name='落座之后开始点菜').first()
end = node_matcher.match(name='多看了隔壁的菜几眼').first()
find_path(graph, start, end)

relationships = graph.run(
    "MATCH path = (startNode)-[*]->(endNode) WHERE id(startNode) = 1134 AND id(endNode) = 1135 RETURN relationships(path)")

    # ,
    # start_id=start.identity, end_id=end.identity)


####


def main():
    # 1、随机选择两个节点，获取节点ID
    nodes = list(graph.run("MATCH (n) RETURN n"))
    node1 = random.choice(nodes)
    node2 = random.choice(nodes)
    while node1 == node2:
        node2 = random.choice(nodes)
    # node1_id = node1["self"]
    # node2_id = node2["self"]

    # 2、找到节点之间的路径并打印
    # paths = find_path(graph, node1_id, node2_id)
    paths = find_path(graph, node1, node2)
    # for path in paths:
    #     print(f"Path from {node1_id} to {node2_id}: {path}")


while True:
    main()

