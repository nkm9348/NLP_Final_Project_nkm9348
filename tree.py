import re

class TreeNode:
    def __init__(self, word, pos, word_cnt, parent=None):
        self.word = word
        self.pos = pos
        self.word_cnt = word_cnt
        self.parent = parent
        self.children: list[TreeNode] = []

def build_tree(parse_tree: str):
    nodes: list[TreeNode] = [] # Stack
    _parse_tree = [x for x in re.split("(\(|\)|\s)", parse_tree) if x.strip() != ""]
    i = 0
    cnt = 0
    while i < len(_parse_tree) - 1:
        element = _parse_tree[i]
        if element == "(":
            pos = _parse_tree[i + 1]
            word = _parse_tree[i + 2]
            word_cnt = cnt
            if pos == "(" or pos == ")" or word == ")":
                raise Exception("Invalid tree")
            if word == "(":
                word = None
                word_cnt = -1
                i += 1
            else:
                i += 2
                cnt += 1

            if word is not None and len(word) > 2 and word[0] == "|" and word[-1] == "|": # Convert |.| to ., etc
                word = word[1:-1]
                if len(pos) > 2 and pos[0] == "|" and pos[-1] == "|":
                    pos = pos[1:-1]
            elif word == "-LRB-":
                word = "("
                pos = "("
            elif word == "-RRB-":
                word = ")"
                pos = ")"

            node = TreeNode(word, pos, word_cnt, nodes[-1] if len(nodes) > 0 else None)
            if len(nodes) > 0:
                nodes[-1].children.append(node)

            nodes.append(node)

        elif element == ")":
            nodes.pop()

        i += 1

    assert len(nodes) == 1
    root = nodes[0]

    return root


def find_leaves(root: TreeNode, cur: list[TreeNode]):
    if root.word is not None:
        cur.append(root)
    for child in root.children:
        find_leaves(child, cur)

    return cur

def find_path_to_node(node: TreeNode, root: TreeNode, cur: list[TreeNode]):
    cur.append(root)

    if root == node:
        return cur.copy()
    
    if root.children is not None:
        for child in root.children:
            t = find_path_to_node(node, child, cur)
            if t is not None:
                return t

    cur.pop()

    return None

def find_path_between_nodes(src: TreeNode, dest: TreeNode, root: TreeNode):
    path1 = find_path_to_node(src, root, [])
    path2 = find_path_to_node(dest, root, [])
    i = 0
    while i < len(path1) and i < len(path2):
        if path1[i] == path2[i]:
            i += 1
        else:
            break
    res = path1[-1].pos
    for j in range(len(path1) - 2, i - 1, -1):
        res += "↑" + path1[j].pos
    if i < len(path1):
        res += "↑" + path1[i - 1].pos
    for j in range(i, len(path2)):
        res += "↓" + path2[j].pos

    return res
