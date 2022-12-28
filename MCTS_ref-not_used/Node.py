from typing import List
from typing_extensions import Literal

class Node(object):
    '''
    # Node
    A node class which contains needed arguments.
    '''
    def __init__(
        self,
        parent = None,
        color: Literal[1, -1] = -1,
        depth: int = 0,
        move: tuple = ()
    ) -> None:
        '''
        Initialize a node.
        ## Parameters\n
        parent: Node, optional
            Parent of this node. Default value is `None` which means it's a root node.
        color: Literal[1, -1], optional
            Color of the stone just played. `1` for black and `-1` for white.
        depth: int, optional
            Depth of this node. Default value is 0.
        move: tuple, opitonal
            Coordinate of current move. Default value is an empty tuple
            which means there is no stone on the board.
        '''
        self.parent = parent
        self.color = color
        self.depth = depth
        self.move = move
        self.value = 0
        self.visits = 0
        self.children: List[Node] = []
        return