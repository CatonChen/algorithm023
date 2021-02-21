# WEEK07学习笔记
### 基础知识
#### Trie树（字典树）
字典树，又称键树、单词查找树。是一种树形数据结构，典型应用场景是统计和排序大量字符串，常见于搜索引擎的文本词频统计。
它的**优点是利用字符串的公共前缀来查询单词，最大限度的减少无谓的字符串比较，查询效率高于哈希表**。
##### 字典树的三个特点
- 根节点不保存字符；根节点外的每个节点不保存完整的单词，只保存一个字符。
- 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串。
- 每个节点的所有子节点保存的字符都不相同。
##### 字典的核心思想
- 空间换时间。
- 利用字符串的公共前缀特性降低查询时间开销达到提高查询效率的目的。
##### 字典树代码模板
**leetcode代表题：208实现前缀树** ref:[https://leetcode-cn.com/problems/implement-trie-prefix-tree/][1]，该题实现了字典树的新增、查找和前缀三个功能。
题解代码即字典树代码模板，如下所示：
```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.end_of_word = '#'

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            if char not in node:
                return False
            else:
                node = node[char]
        return self.end_of_word in node

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            else:
                node = node[char]
        return True

```
附一个测试用例参考
```python
测试用例:["Trie","insert","search","startsWith"]
		[[],["apple"],["apple"],["app"]]
		测试结果:[null,null,true,true]
		期望结果:[null,null,true,true]
		stdout:
		insert:apple
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		insert done
		search:apple
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		{'p': {'p': {'l': {'e': {'#': '#'}}}}}
		{'p': {'l': {'e': {'#': '#'}}}}
		{'l': {'e': {'#': '#'}}}
		{'e': {'#': '#'}}
		{'#': '#'}
		True
		search done
		start with:app
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		{'l': {'e': {'#': '#'}}}
		start done
```

#### 并查集（Disjoint Sets）
并查集是一种树型数据结构，用来处理一些不相交集合的合并与查询问题。常常在使用中以森林来表示。
1. 并查集的基础概念
	1. 一种数据结构。
	2. 并查集三个字，一个字一个含义。
		1. 并（union），代表合并。
		2. 查（find），代表查找。
		3. 集（set），代表这是一个以字典为基础的数据结构，它的功能是合并集合中的元素，查找集合中的元素。
	3. 典型应用是有关**连通分量**的问题。
	4. 解决单个问题（合并，查找，添加）的**时间复杂度都是O(1)**。
	5. 应用在**在线算法**中。
2. 并查集的实现模板
	1. **数据结构**
		1. 与树类似，但与树相反。**树记录的是节点的子节点；并查集记录的节点的父节点**。
		2. 如果节点之间是连通的，那么它们在同一个集合里，它们的祖先是相同的。
	2. **新增**：新节点添加到并查集里时，它的父节点为空。
	3. **合并**：如果两个节点是连通的，就要把它们合并，这里谁做父节点没有区别，因为祖先是相同的。
	4. **是否连通**：判断两个节点是否处于同一个连通分量，就**判断它们的祖先是否相同**。
	5. **查找**
		1. 如果节点的父节点不为空，就不断迭代，直到最上（外）层的节点的父节点为空，该节点就是祖先。
		2. 优化（路径压缩）：**如果树很深，退化成链表，那么就做路径压缩，把树的深度固定为二**。这么做的可行性是，因为并查集只记录节点的连通关系，而节点相互连通只需要有一个公共祖先即可。路径压缩可以用递归或迭代。
	6. **并查集代码模板**
```python
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
    
    def find(self,x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] != None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
         
        return root
    
    def merge(self,x,y):
        """
        合并两个节点
        """
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self,x,y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)
    
    def add(self,x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
```
3. leetcode代表题：547.省份数量 ref:[https://leetcode-cn.com/problems/number-of-provinces/][2]

#### 高级搜索
##### 剪枝
深度优先搜索（DFS）和广度优先搜索（BFS），其本质是对一颗状态树的深度、广度优先遍历。
对于一颗树，以深度优先遍历为例。如果从根节点开始深度搜索，若要完全遍历，则需要搜索所有节点。但在解决实际问题的过程中，往往可以发现有一些点和它们的子节点根本不符合题意，压根儿没有搜索的必要。那么在算法中加入一个判断条件，使得在这颗子树在搜索时不会进入，从而优化了时间复杂度。这个优化技巧，如同在树上砍掉冗余的枝条，所以叫“剪枝”。
###### 常见的剪枝方式
- **可行性剪枝**：指在当前情况与题意不符时，以及可以推导出后续情况都与题意不符，那么就进行剪枝，直接把这种情况及后续情况判负，返回。即：不可行，就返回。
- **排除等效冗余**：指当几个分支具有完全相同效果时，选择其中一个即可。即：都可以，选一个。
- **最优性剪枝**：指在用搜索方法解决最优化问题时的一种常用剪枝。就是搜索到一半时，发现相比已经搜索到的最优解，继续搜索不到更优解，那么停止搜索，进行回溯。即：有比较，选最优。
- **顺序剪枝**：普遍来说，搜索是不固定的，对于一个问题，算法可以进入搜索树的任意一个节点。假如需要搜索最小值，但非要从保存最大值的节点开始，那么可能要搜索到最后才有解；但是如果一开始从保存最小值的节点开始，那么可能立即得到解。这是顺序剪枝的一个应用。一般来说，有单调性存在的搜索问题，可以结合贪心算法，进行顺序剪枝。即：有顺序，按题意。
- **记忆化**：等同于记忆化搜索，搜索的一种分支。就是记录搜索的每一种状态，当重复搜索到相同状态时，直接返回已知解。即：搜重来，直接跳。
原文ref：[https://www.cnblogs.com/fusiwei/p/11759489.html][3]
###### leetcode代表题目
- 70.爬楼梯：[https://leetcode-cn.com/problems/climbing-stairs/][4]，重复性搜索、记忆化搜索。
- 22.括号生成：[https://leetcode-cn.com/problems/generate-parentheses/][5]，可行性剪枝。
- 51.N皇后：[https://leetcode-cn.com/problems/n-queens/][6]，回溯+可行性剪枝。

##### 回溯
采用试错的思想，尝试分步解决一个问题。在分步过程中，当发现可能现有的分步答案不能得到有效的正确解时，可能返回上一步或上几步，重新尝试用其他分步再次寻找有效解。这里的分步过程，就是分治。
回溯通常用递归方法实现，时间复杂度可能到指数级复杂度，在重复寻求答案过程后可能存在：
- 有一个有效的解
- 没有求得任何解

###### 代码模板
DFS代码-递归写法
```python
visited = set()
def dfs(node, visited):
  if node in visited: # terminator
      # already visited
	return
	visited.add(node)
	# process current node here.
	...
	for next_node in node.children(): 
		if not next_node in visited:
		dfs(next_node, visited)
```

DFS代码-非递归写法（手工维护栈）
```python
def DFS(self, tree):
	if tree.root is None:
		return []
	visited, stack = [], [tree.root]
	while stack:
	node = stack.pop() 
	visited.add(node)
	process (node)
	nodes = generate_related_nodes(node) 
	stack.push(nodes)
    # other processing work
	...
```

BFS代码
```python
def BFS(graph, start, end):
	queue = [] queue.append([start]) 
	visited.add(start)
	while queue:
		node = queue.pop() 
		visited.add(node)
		process(node)
		nodes = generate_related_nodes(node) 
		queue.push(nodes)
```

##### 双向BFS
leetcode代表题目
127.单词接龙：[https://leetcode-cn.com/problems/word-ladder/][7]
以此题为例，在单向BFS的基础上进行优化。与单向BFS不同在于，从begin和end两边同时开始搜索，向两者之间的共同部分靠拢。当两个搜索出现重叠的公共部分时，即说明搜索成功，反之没有可行解。
单向BFS如同水波纹，一层层向外扩散，那双向BFS则如同两侧同时进行水波纹扩散，而题解就在两侧水波纹的交汇处。
###### 单向BFS示意图
![]()
###### 双向BFS示意图
![]()
###### 127.单词接龙（单向和双向BFS算法学习总结）
1. 单向BFS
	1. 算法思路
		1. 一开始判断下endWord是否在wordList中，不在可以直接返回0，因为不存在可能的解。
		2. **将wordList处理成set集合，有利于单词转换中对wordList的查询，降低时间复杂度**。（很重要）
		3. 创建一个队列queue，初始将beginWord和次数1加入，queue.append(beginWord,1)。
		4. 创建一个set集合visited，用于保存已访问过的单词，初始将beginWord加入。
		5. 当queue不为空时，循环遍历队列；取出队列首端元素赋给cur和count两个变量（cur,count=queue.popleft()），进行如下操作：
			1. 当cur等于endWord时，表示已匹配到目标单词，返回count。
			2. 当cur不等于endWord时，继续做以下操作：
				1. 遍历cur的每个字母，同时用26个字母逐个替换它，生成tmp单词。
				2. 当tmp单词在wordList中有匹配对象且不在visited里时，将tmp单词加入队列，次数count+1；同时将tmp单词放入visited集合，记录已被使用。
		6. 当队列为空后，如果没有匹配到目标单词，返回0，表示没有接龙成功。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M表示单词数量，N表示单词字母数。
		2. 空间复杂度：O(N)，N表示wordList和visited集合使用的空间。
	3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # wordlist——>set
        st = set(wordList)
        # endword不在st中，返回0
        if endWord not in st:
            return 0
        # 初始化队列queue和visited
        from collections import deque
        queue = deque()
        queue.append((beginWord, 1))
        visited = set()
        visited.add(beginWord)
        # 遍历队列
        while queue:
            cur, count = queue.popleft()
            # 找到目标单词，返回次数
            if cur == endWord:
                return count
            # 没找到目标，继续处理
            for i in range(len(cur)):
                # 单词字母替换
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    tmp = cur[:i] + j + cur[i + 1:]
                    # tmp未被使用过，且在wordlist中
                    if tmp in st and tmp not in visited:
                        queue.append((tmp, count + 1))
                        visited.add(tmp)
        # 没有可能解，返回0
        return 0
```

2. 双向BFS
	1. 算法思路
		1. 在单向BFS上进行优化处理。与单向BFS不同之处在于，对beginWord和endWord分别做了队列和记录使用的集合，lqueue、lvisited、rqueue和rvisited。
		2. **以层为单位记录接龙次数**。只有当队列里当前元素都处理完了，才对接龙次数加一，这点与单向BFS不同。
		3. 当lqueue和rqueue均不为空时，先判断哪个队列较短，后续对较短的队列进行操作即可。假设lqueue队列为元素少的队列：
			1. 从lqueue中取出首端元素cur，当cur在rvisited中已经存在，即说明接龙成功，返回层数。
			2. 当cur不在rvisited中时，仍然用26字母逐个替换cur单词的字母，生成tmp。
			3. 当tmp不在lvisited中，又在wordList中时，将tmp加入lvisited和lqueue中。
		4. 如最终没有接龙成功，返回0。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M为单词数，N为单词字母个数。
		2. 空间复杂度：O(N)，队列和集合需要利用额外空间。
	3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # wordlist——>set
        st = set(wordList)
        # endword不在st中，返回0
        if endWord not in st:
            return 0
        # 初始化队列queue和visited
        from collections import deque
        lqueue = deque()
        lqueue.append(beginWord)
        lvisited = set()
        lvisited.add(beginWord)
        rqueue = deque()
        rqueue.append(endWord)
        rvisited = set()
        rvisited.add(endWord)
        step = 0
        # 遍历队列
        while lqueue and rqueue:
            # 找出元素少的队列
            if len(lqueue) > len(rqueue):
                lqueue, lvisited, rqueue, rvisited = rqueue, rvisited, lqueue, lvisited
            # 次数+1
            step += 1
            # 对短队列处理每一个元素
            for k in range(len(lqueue)):
                cur = lqueue.popleft()
                # 在rvisited找到目标单词，返回次数
                if cur in rvisited:
                    return step
                # 没找到目标，继续处理
                else:
                    for i in range(len(cur)):
                        # 单词字母替换
                        for j in 'abcdefghijklmnopqrstuvwxyz':
                            tmp = cur[:i] + j + cur[i + 1:]
                            # tmp未被使用过，且在wordlist中
                            if tmp in st and tmp not in lvisited:
                                lqueue.append(tmp)
                                lvisited.add(tmp)
        # 没有可能解，返回0
        return 0
```

##### 启发式搜索 Heuristic Search (A\*)
启发式搜索(Heuristically Search)又称为有信息搜索(Informed Search)，它是利用问题拥有的启发信息来引导搜索，达到减少搜索范围、降低问题复杂度的目的，这种利用启发信息的搜索过程称为启发式搜索。
启发式策略可以通过指导搜索向最有希望的方向前进，降低了复杂性。通过删除某些状态及其延伸，启发式算法可以消除组合爆炸，并得到令人能接受的解(通常并不一定是最佳解)。
###### 启发式搜索代码模板
基于BFS代码模板而来，**但与BFS不同的是，队列使用了优先队列**。因为使用了优先队列，所以每次队列弹出的元素是根据优先级从高往低进行的，优先级越高的元素意味着得到解的可能性越高。决定元素优先级高低的函数叫启发式函数，又叫估价函数。
```python
def AstarSearch(graph, start, end):	pq = collections.priority_queue() # 优先级 —> 估价函数
	pq.append([start]) 	
	visited.add(start)	
	while pq: 		
		node = pq.pop() # can we add more intelligence here ?
		visited.add(node)		
		process(node) 		
		nodes = generate_related_nodes(node)    
		unvisited = [node for node in nodes if node not in visited]		
		pq.push(unvisited)
```
###### 估价函数
用于评价节点重要性的函数，称为估计函数。
公式：**f(x)=g(x)+h(x)**
1. g(x)为从初始节点到节点x的实际代价。
2. h(x)为从节点x到目标节点的最优路径的估计代价。启发式信息主要体现在h(x)中，其形式根据问题的特性来定。
	- 启发式函数: h(x)，它用来评价哪些结点最有希望的是一个我们要找的结点，h(x) 会返回一个非负实数,也可以认为是从结点x的目标结点路径的估计成本。
	- 启发式函数是一种告知搜索方向的方法。它提供了一种明智的方法来猜测哪个邻居结点会导向一个目标。
###### A\*search
重点在于估计函数的定义。
###### leetcode代表题目
- 1091.二进制矩阵中的最短路径：[https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/][8]
- 773.滑动谜题：[https://leetcode-cn.com/problems/sliding-puzzle/][9]
###### 资料参考
- 启发式搜索——A\*算法 ref:[https://www.cnblogs.com/ISGuXing/p/9800490.html][10]
- 相似度测量方法 ref:[https://dataaspirant.com/five-most-popular-similarity-measures-implementation-in-python/][11]
- 8puzzles解法比较 ref:[https://zxi.mytechroad.com/blog/searching/8-puzzles-bidirectional-astar-vs-bidirectional-bfs/][12]

### 本周leetcode练习总结
#### 74.搜索二维矩阵（本周复习）
1. 矩阵一维化+二分查找
	1. 一维化
		1. 因矩阵又两个特点：**a.每一行是生序排列，b.每一行的第一个元素大于前一行的最后一个元素**。故将二维矩阵转换为一维数组后，该数组是个生序排列的数组。
		2. 一维化方法：**nums=[ i for row in matrix for i in row]**
	2. 二分查找
		1. 对一维生序数组寻找目标值，适用二分查找的方法。令left=0, right=len(nums)-1；当left\<=right时，mid=(left+right)/2。
		2. 当nums[mid]=target时，则返回True
		3. 当nums[mid]\>target时，令right=mid-1；否则令left=mid+1
		4. 如果查找没有结果，则返回False。
	3. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵所有元素，所以有m行n列，即O(m\*n)；二分查找的时间复杂度一般为O(logn)。
		2. 空间复杂度：O(n)，利用了一个数组空间，n为m\*n的乘积
	4. 题解代码
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 特例
        if not matrix:
            return False
        # 矩阵一维化
        nums = [i for row in matrix for i in row]
        # print(nums)
        # 二分查找
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return False
```

2. 线性扫描+二分查找
	1. 因矩阵的特点，**可以让target与矩阵每行最后一个元素比较大小，如果target\<matrix[i][-1]，则说明target在matrix[i]中，那么后续通过二分查找该行即可**。
	2. 在matrix[i]中进行二分查找，如果匹配到target则返回true，否则返回false。
	3. 复杂度分析
		1. 时间复杂度：O(logn+m)，二分查找为O(logn)，m为矩阵行数
		2. 空间复杂度：O(1)，常量变量
	4. 题解代码
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 特例
        if not matrix:
            return False
        # 线性扫描
        index = -1
        for i in range(len(matrix)):
            if target <= matrix[i][-1]:
                index = i
                break
        if index == -1:
            return False  # target不在矩阵里
        else:  # 二分查找
            left, right = 0, len(matrix[index]) - 1
            while left <= right:
                mid = (left + right) // 2
                if matrix[index][mid] == target:
                    return True
                elif matrix[index][mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return False
```

#### 589.N叉树的前序遍历（本周复习）
1. 递归法
	1. 定义递归函数，进行递归工作：
		1. 递归终止：**当节点为空时，返回；否则将节点值加入结果**
		2. **当该节点有子节点时，将子节点依次调用递归函数进行递归工作**
	2. 对根节点进行递归工作，返回最终结果。
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历树的所有节点
		2. 空间复杂度：O(n)，递归工作时系统栈
	4. 题解代码
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []

        def dfs(root):
            if not root:
                return
            res.append(root.val)
            for child in root.children:
                dfs(child)

        dfs(root)
        return res
```

2. 迭代法（手工维护栈）
	1. 定义一个数组stack，初始将根节点root压入stack。
	2. 当stack不为空时，pop节点node，将其值加入结果
	3. 当node存在子节点时，逆序将子节点压入stack
	4. 当stack为空时，返回最终结果
	5. 复杂度分析
		1. 时间复杂度：O(n)，遍历树的所有节点
		2. 空间复杂度：O(n)，stack保存了所有节点
	6. 题解代码
```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        if not root:
            return res

        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            for n in node.children[::-1]:
                stack.append(n)

        return res
```

#### 208.实现Trie（字典树）
1. 算法思路
	1. trie定义为一个字典（root=\{}），并设置单词结束时的标识符（end\_of\_word=‘#’）。
	2. insert方法
		1. 初始将节点node初始化为一个trie
		2. 遍历单词中每个字母，将字母依次加入到节点node中，这里类似递归工作，当前字母作为key，后续所有字母作为value。
		3. 当单词遍历完毕后，在node中添加结束标识符（重要！）
		4. 例子：插入apple单词，最终trie为:
			{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		5. 复杂度分析
			1. 时间复杂度：O(n)，遍历单词的所有字母
			2. 空间复杂度：O(n)，保存单词的所有字母
	3. search&prefix方法
		1. 将node初始化为root
		2. 遍历单词中的每个字母，如果存在字母在node中找不到的情况，返回false，说明单词不在trie中。
		3. 否则，node不断被更新trie中为当前字母的value，直到遍历完成。
		4. 最终判断剩下的node
			1. 如果是search，node中含有结束标识符‘#’，则返回true，否则返回false，说明trie中没有匹配的单词
			2. 如果是prefix，只要单词中的每个字母都能在trie中找到，就返回true
		5. 复杂度分析
			1. 时间复杂度：O(n)，无论是search或prefix，都要遍历单词的所有字母
			2. 空间复杂度：O(1)
	4. 题解代码
```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.end_of_word = '#'

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        # print('insert:'+word)
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word
        # print(self.root)
        # print('insert done')


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        # print('search:'+word)

        node = self.root
        # print(node)
        for char in word:
            if char not in node:
                # print('search false')
                return False
            else:
                node = node[char]
                # print(node)
        # print(node)
        # print(self.end_of_word in node)
        # print('search done')
        return self.end_of_word in node

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        # print('start with:'+prefix)
        node = self.root
        # print(node)
        for char in prefix:
            if char not in node:
                # print('start with false')
                return False
            else:
                node = node[char]
        # print(node)
        # print('start done')
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```
5. 测试用例参考
测试用例:["Trie","insert","search","startsWith"]
		[[],["apple"],["apple"],["app"]]
		测试结果:[null,null,true,true]
		期望结果:[null,null,true,true]
		stdout:
		insert:apple
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		insert done
		search:apple
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		{'p': {'p': {'l': {'e': {'#': '#'}}}}}
		{'p': {'l': {'e': {'#': '#'}}}}
		{'l': {'e': {'#': '#'}}}
		{'e': {'#': '#'}}
		{'#': '#'}
		{'#': '#'}
		True
		search done
		start with:app
		{'a': {'p': {'p': {'l': {'e': {'#': '#'}}}}}}
		{'l': {'e': {'#': '#'}}}
		start done

#### 628.三个数的最大乘积（本周复习）
1. 排序法
	1. 首先对数组nums排序，之后考虑两种情况
		1. 假设排序后的最大的三个数的乘积可能最大，最大值为a
		2. 假设数组中存在负数的可能，那么最小的两个负数\*最大的一个正数，其乘积可能最大，最大值为b
	2. 最终返回a和b中的最大值即可
	3. 复杂度分析：
		1. 时间复杂度：O(nlogn)，排序耗时
		2. 空间复杂度：O(n)
	4. 题解代码
```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        # 排序
        nums.sort()
        a = nums[-3] * nums[-2] * nums[-1]
        b = nums[0] * nums[1] * nums[-1]
        return max(a, b)
```

#### 200.岛屿数量（本周复习）
1. 深度优先搜索（递归）
	1. 算法思路
		1. 因为陆地是1，水是0，且陆地总是被水包围着的。那么当遇到岛屿时，做两件事
			1. **岛屿数量+1**
			2. **同时将当前岛屿覆盖的陆地1变为水0，然后寻找下一个岛屿。这里的1，除了当前矩阵单元格的1外，还有相邻的1，及间接相邻的1（即连续相邻的1）**。
		2. 当整个矩阵都从1变为0后，返回记录的岛屿数量即可。
	2. 当遇到1时，进行递归工作，具体如以下
		1. 递归终止条件有两点：
			1. 当前单元格坐标（x,y）超出矩阵范围
			2. 当前单元格不是1
		2. 递归工作有两点：
			1. 将矩阵当前单元格（x,y）置为0
			2. 对当前单元格的**相邻单元格(x-1,y),(x+1,y),(x,y+1),(x,y-1)**进行同样的递归工作
	3. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵
		2. 空间复杂度：O(m\*n)，最差递归m\*n次
	4. 题解代码
```python
class Solution:
    # dfs
    def dfs(self, grid, i, j):
        # 终止条件
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return
            # 当前处理
        grid[i][j] = '0'
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)

    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        # grid为空
        if not grid:
            return count
        # 遍历grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1

        return count
```

665.非递减序列（本周复习）
1. 迭代遍历
	1. 算法思路
		1. 因非递减序列，总是满足**num[i]\<=nums[i+1]**，那么它是一个单调递增序列。那么当出现**nums[i]\<nums[i-1]**时，就破坏了它的有序性，这时考虑将其修复的可能方法，同时记录修复次数。最终修复次数\<=1，即满足题目要求的最多1次，返回true，否则返回false。
		2. 修复方法的场景有三种
			1. 当i=1时，将nums[i-1]修改为nums[i]，同时count+1。
			2. 当i\>1时，且nums[i]\>=nums[i-2]，则将nums[i-1]修改为nums[i]，同时count+1。
			3. 当i\>1时，且nums[i]\<nums[i-2]，则将nums[i]修改为nums[i-1]，同时count+1。
		3. 最终返回count\<=1的布尔值即可
	2. 复杂度分析
		1. 时间复杂度：O(n)，遍历一次数组
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                count += 1
                if i == 1 or nums[i] >= nums[i - 2]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]
        return count <= 1
```

#### 860.柠檬水找零（本周复习）
1. 贪心算法
	1. 算法思路
		1. 设定两个变量five=0, ten=0。接下来在算法内实现有几种可能的情况，最终根据是否可以实现返回布尔值。
			1. 一开始获得的钱是10或20，不能找零，返回false。
			2. 获得的钱是5，则five+1
			3. 获得的钱是10，且five\>0，则ten+1，five-1。
			4. 获得的钱是20，（优1）如果ten\>0 & five\>0，则ten-1,five-1；（优2）如果five\>2，则five-3
			5. 当遇到以上任一条件不满足时，返回false；否则顺利完成，返回true。
	2. 复杂度分析
		1. 时间复杂度：O(n)，遍历整个数组
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five = ten = 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five > 0:
                    ten += 1
                    five -= 1
                else:
                    return False
            else:
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five > 2:
                    five -= 3
                else:
                    return False
        return True
```

#### 547.省份数量
1. 并查集
	1. 并查集的基础概念
		1. 一种数据结构。
		2. 并查集三个字，一个字一个含义。
			1. 并（union），代表合并。
			2. 查（find），代表查找。
			3. 集（set），代表这是一个以字典为基础的数据结构，它的功能是合并集合中的元素，查找集合中的元素。
		3. 典型应用是有关**连通分量**的问题。
		4. 解决单个问题（合并，查找，添加）的**时间复杂度都是O(1)**。
		5. 应用在**在线算法**中。
	2. 并查集的实现模板
		1. **数据结构**
			1. 与树类似，但与树相反。**树记录的是节点的子节点；并查集记录的节点的父节点**。
			2. 如果节点之间是连通的，那么它们在同一个集合里，它们的祖先是相同的。
		2. **初始化**：新节点添加到并查集里时，它的父节点为空。
		3. **合并**：如果两个节点是连通的，就要把它们合并，这里谁做父节点没有区别，因为祖先是相同的。
		4. **两节点是否连通**：判断两个节点是否处于同一个连通分量，就**判断它们的祖先是否相同**。
		5. **查找祖先**
			1. 如果节点的父节点不为空，就不断迭代，直到最上（外）层的节点的父节点为空，该节点就是祖先。
			2. 优化（路径压缩）：**如果树很深，退化成链表，那么就做路径压缩，把树的深度固定为二**。这么做的可行性是，因为并查集只记录节点的连通关系，而节点相互连通只需要有一个公共祖先即可。路径压缩可以用递归或迭代。
		6. **并查集代码模板**
```python
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
    
    def find(self,x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] != None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
         
        return root
    
    def merge(self,x,y):
        """
        合并两个节点
        """
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self,x,y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)
    
    def add(self,x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
```
3. 算法思路
	1. 省份数量即计算城市节点之间的连通分量的数量。
		2. 在并查集模板中添加一个变量记录集合的数量
			1. 初始化时+1
			2. 合并时-1
		3. 最终返回记录集合的数量的变量即可。
	4. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵，并查集添加、合并、查找的时间复杂度均为O(1)
		2. 空间复杂度：O(n)
	5. 题解代码
```python
class unionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
        # 记录集合数量
        self.num_of_set = 0

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
            # 集合数量+1
            self.num_of_set += 1

    def find(self, x):
        """
        查找节点
        """
        root = x
        while self.father[root] != None:
            root = self.father[root]

        """
        路径压缩
        """
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def merge(self, x, y):
        """
        合并两个节点
        """
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y
            # 集合数量-1
            self.num_of_set -= 1

    # def isconnected(self,x,y):
    #     """
    #     判断两个节点是否相连
    #     """
    #     return self.find(x)==self.find(y)


class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        # 初始化并查集
        uf = unionFind()
        for i in range(len(M)):
            # print('i: ' + str(i))
            uf.add(i)  # 添加节点
            for j in range(i):
                # print('j: ' + str(j))
                # print('i: ' + str(i) + ' j: ' + str(j) + ' M[i][j]: ' + str(M[i][j]))
                if M[i][j] == 1:  # M[i][j]有效
                    uf.merge(i, j)

        return uf.num_of_set
```

#### 874.模拟行走机器人（本周复习）
1. 贪心算法
	1. 算法思路
		1. 创建一个字典，用于保存当前方向的**x移动、y移动、当前左侧和当前右侧**，如当前方向dir=‘up’，则dict=\{‘up’:[0,1,’left’,’right’]}。
		2. 将障碍物坐标元组化（tuple），再集合化（set）。**这里set很重要，否则增加时间复杂度**。
		3. 进行指令模拟
			1. 当command==-1（右转）or-2（左转）时，更新dir
			2. 当command\>0时，不断迭代，模拟行走。
				1. 当遇到障碍物坐标时，循环break，停止行走。
				2. 否则返回欧式距离平方根x^2+y^2
	2. 复杂度分析
		1. 时间复杂度：O(n)，指令行走的步数n和查询障碍物坐标的时间复杂度O(1)。
		2. 空间复杂度：O(n)
	3. 题解代码
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 初始化方向字典
        dirs = {
            'up': [0, 1, 'left', 'right'],
            'right': [1, 0, 'up', 'down'],
            'left': [-1, 0, 'down', 'up'],
            'down': [0, -1, 'right', 'left']
        }
        # 障碍物坐标set
        obstaclesset = set(map(tuple, obstacles))
        # 初始化x,y,res,dir
        x = y = res = 0
        dir = 'up'
        # 模拟行走
        for cmd in commands:
            if cmd == -2:  # 左转
                dir = dirs[dir][2]
            elif cmd == -1:  # 右转
                dir = dirs[dir][3]
            else:
                for i in range(cmd):
                    if (x + dirs[dir][0], y + dirs[dir][1]) in obstaclesset:
                        break
                    else:
                        x += dirs[dir][0]
                        y += dirs[dir][1]
                        res = max(res, x ** 2 + y ** 2)

        return res
```
4. 左转与右转的另一种实现
	1. 初始朝北，x=y=0，步数dx=0,dy=1。
		2. 朝北时进行左转，dx,dy=-dy,dx。因为左转后，移动时x轴-1，y轴不变，
		3. 朝北时进行右转，dx,dy=dy,-dx。因为右转后，移动时x轴+1，y轴不变。
	5. 题解代码
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 障碍物坐标set
        obstaclesset = set(map(tuple, obstacles))
        # 初始化x,y,res
        x = y = res = 0
        # 初始化步数
        dx, dy = 0, 1
        # 模拟行走
        for cmd in commands:
            if cmd == -2:  # 左转
                dx, dy = -dy, dx
            elif cmd == -1:  # 右转
                dx, dy = dy, -dx
            else:
                for i in range(cmd):
                    if (x + dx, y + dy) in obstaclesset:
                        break
                    else:
                        x += dx
                        y += dy
                        res = max(res, x ** 2 + y ** 2)

        return res
```

#### 119.杨辉三角II
1. 滚动数组
	1. 算法思路
		1. 由杨辉三角的特点得知，第i行第j列=第i-1行第j-1列+第i-1行第j列，即公式**res[i][j]=res[i-1][j-1]+res[i-1][j]**。
		2. 创建一个二维数组res，进行滚动迭代更新res[i][j]的值。注意两点：a.将res中每个值都初始化为1；b.数组由0开始。
	2. 复杂度分析
		1. 时间复杂度：O(n^2)
		2. 空间复杂度：O(n\*(n+1)/2)
	3. 题解代码
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [[1 for j in range(i + 1)] for i in range(rowIndex + 1)]
        # print(res)
        for i in range(2, rowIndex + 1):
            for j in range(1, i):
                res[i][j] = res[i - 1][j - 1] + res[i - 1][j]
        # print(res)
        return res[-1]
```
4. 空间复杂度优化为O(n)（动态规划）
	1. 创建一维数组res，长度为rowIndex+1。
		2. 从右往左遍历，res[j]=res[j]+res[j-1]。
		3. 为何不从左往右遍历？**因为从左往右遍历，左边的元素已经被更新为第i行的元素了，但res[j]需要第i-1行的元素**。
		4. 题解代码
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [1] * (rowIndex + 1)
        # print(res)
        for i in range(2, rowIndex + 1):
            for j in range(i - 1, 0, -1):
                res[j] += res[j - 1]
                # print(res)
        # print(res)
        return res
```

2. 通项公式（动态规划）
	1. 通项公式：res[i][j]=res[i][j-1]\*(i-j+1)/j
	2. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)
	3. 题解代码
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [1] * (rowIndex + 1)
        for i in range(1, len(res) - 1):
            # print(res[i - 1])
            # print((rowIndex - i + 1) // i)
            res[i] = res[i - 1] * (rowIndex - i + 1) // i
        return res
```

#### 888.公平的糖果交换（本周复习）
1. 哈希表
	1. 算法思路
		1. 设糖果棒A的总大小为sumA和单个大小为x，糖果棒B的总大小为sumB和单个大小为y；则有**sumA-x+y=sumB+x-y**才能达到公平交换的目的。
		2. sumA-x+y=sumB+x-y，化简得，**x=y+(sumA-sumB)/2**。那么对于B中的任意一个y，只要A中的一个x满足该公式即可得到一组可行解(x,y)。
		3. 为了快速查询A，首先用set将A哈希化。
	2. 复杂度分析
		1. 时间复杂度：O(n+m)，数组A的长度n+数组B的长度m
		2. 空间复杂度：O(n)，保存了数组A的n个元素
```python
class Solution:
    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        sumA, sumB = sum(A), sum(B)
        diff = (sumA - sumB) // 2
        setA = set(A)
        for y in B:
            x = y + diff
            if x in setA:
                return [x, y]
```

#### 130.被围绕的区域
1. 深度优先搜索（DFS）
	1. 算法思路
		1. 根据题目，得知矩阵内有三种元素
			1. 字母X
			2. 被字母X包围的字母O
			3. 没有被字母X包围的字母O
		2. 根据解释，任何边界上的字母O都不会被包围。那么所有不被包围的O都是直接或间接与边界上的O相连。利用这个性质：从边界上的O开始将所有直接与间接相连的O都做好标记。
		3. 最后遍历矩阵，对于每个字母：
			1. 如果标记过，则说明该字母为没有被X包围的O，将其还原成O。
			2. 如果没有标记过，则说明该字母为被X包围的O，将其变为X。
		4. 第2点中，标记字母O的递归工作，与题目200.岛屿数量的递归方法类似。
	2. 复杂度分析
		1. 时间复杂度：O(m\*n)，m、n分别矩阵的行和列。
		2. 空间复杂度：O(m\*n)，递归工作时生成的系统栈。
	3. 题解代码
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 特例
        if not board:
            return None

        # 矩阵的行列
        m, n = len(board), len(board[0])
        # print(m,n)
        # dfs
        def dfs(board, x, y):
            # dfs终止条件
            if not 0 <= x < m or not 0 <= y < n or board[x][y] != 'O':
                return
            board[x][y] = 'A'
            dfs(board, x - 1, y)
            dfs(board, x + 1, y)
            dfs(board, x, y - 1)
            dfs(board, x, y + 1)

        # 遍历矩阵边界
        for i in range(m):
            # print(i)
            dfs(board, i, 0)
            dfs(board, i, n - 1)
        for j in range(n):
            # print(j)
            dfs(board, 0, j)
            dfs(board, m - 1, j)

        #遍历矩阵，标记的还原成O，其余都标为X
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'A':
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
```

2. 广度优先搜索（BFS）
	1. 算法思路与深度优先搜索（DFS）类似，不同之处在于将递归工作改成了维护一个队列，将与矩阵边界上的O直接或间接相连的O都放进了队列中进行标记。队列工作主要分为两步：
		1. 先遍历矩阵边界，将字母O的坐标记录在队列中。
		2. 弹出队列中字母O的坐标，将其进行标记，再遍历该坐标的相邻且在矩阵范围内的字母O的坐标，将其压入队列。不断迭代，直到将直接或间接相邻的字母O的坐标遍历完毕为止。
	2. 复杂度分析与深度优先搜索一致
	3. 题解代码
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        from collections import deque

        # 特例
        if not board:
            return None

        # 矩阵的行列
        m, n = len(board), len(board[0])
        # print(m,n)
        # bfs
        queue = deque()
        # 先遍历矩阵边界的O
        for i in range(m):
            if board[i][0] == 'O':
                queue.append((i, 0))
            if board[i][n - 1] == 'O':
                queue.append((i, n - 1))

        for j in range(n):
            if board[0][j] == 'O':
                queue.append((0, j))
            if board[m - 1][j] == 'O':
                queue.append((m - 1, j))

        # 在遍历与边界O直接或间接相邻的O
        while queue:
            x, y = queue.popleft()
            board[x][y] = 'A'
            for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                    queue.append((mx, my))

        # 遍历矩阵，标记的还原成O，其余都标为X
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'A':
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
```

3. 并查集
	1. 算法思路
		1. 与深度优先搜索类似，不同在于使用并查集的方法对矩阵内的所有节点做了连通分类。
		2. 并查集实现
			1. 虚拟一个dummy节点，作为没有被X包围的O的公共祖先
			2. 将矩阵内所有字母O添加进并查集内，边界字母O与dummy合并拥有相同的父节点；非边界字母O则有两种情况：
				1. 与边界O相连的，最终也与dummy拥有相同的父节点。
				2. 不与边界O相连的O，会互相连通，将其中一个O的父节点作为公共祖先，但与dummy的不一致。
		3. 最终遍历整个矩阵，判断每个字母的是否与dummy拥有相同的父节点，有则保留O，否则改为X。
	2. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵，并查集的时间复杂度约为O(1)。
		2. 空间复杂度：O(m)，并查集保存空间
	3. 题解代码
```python
# 并查集模板
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}

    def find(self, x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] is not None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def merge(self, x, y):
        """
        合并两个节点
        """
        root_x, root_y = self.find(x), self.find(y)

        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self, x, y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None

        # 初始化矩阵行列
        m, n = len(board), len(board[0])

        # 定义node
        def node(x, y):
            return x * n + y

        # 初始化uf
        uf = UnionFind()
        # 初始化dummy
        dummy = m * n
        uf.add(dummy)
        # print(uf.father)

        # 遍历矩阵检查字母O
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    # 属于矩阵边界的O
                    if i == 0 or j == 0 or j == n - 1 or i == m - 1:
                        # print(node(i,j))
                        uf.add(node(i, j))
                        # 将边界O与dummy连通
                        uf.merge(node(i, j), dummy)
                    else:
                        # 非边界的O
                        for mx, my in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                            if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                                uf.add(node(i, j))
                                uf.add(node(mx, my))
                                # 将相互连通的O进行合并
                                # 与边界O连通的O，最终会连通至dummy
                                uf.merge(node(mx, my), node(i, j))
        #
        # print(uf.father)
        # 遍历矩阵
        for i in range(m):
            for j in range(n):
                uf.add(node(i, j))
                # 与dummy是同一个祖先
                if uf.is_connected(dummy, node(i, j)):
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
```

#### 989.数组形式的整数加法
1. 类型变换
	1. 算法思路
		1. 将A变换成string，再变换成int。
		2. K本身是int，所以两者相加之后，再变换成string，再转成list输出即可。
		3. **类似题目66.加一**
	2. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        return list(str(int(''.join(map(str, A))) + K))
```

2. 逐位相加
	1. 算法思路
		1. A为数组形式取逐位。
		2. K通过对10取余来取逐位，对10整除（地板除）向前进位。
		3. 从低位向高位逐位相加，当和\>10时，将进位计入下一位相加。
		4. 注意两种情况
			1. 数组A的长度小于整数K的位数个数：**遍历完数组A后，K仍\>0，需要对K继续取余并向前进位，直至K=0**。
			2. 数组A的长度大于整数K的位数个数：**因为第2、3点中K通过取余低位相加和整除向前进位，所以最终K=0时，相加结果不变**。
	2. 复杂度分析
		1. 时间复杂度：O(max(n,logk))，n指数组A的长度，logk指对K取余进位的耗时
		2. 空间复杂度：O(n)
	3. 题解代码
```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        n = len(A)
        res = []
        # 逆序迭代，从低位到高位
        for i in range(n - 1, -1, -1):
            tmp = A[i] + K % 10
            # K进位
            K //= 10
            # tmp>=10，K进位
            if tmp >= 10:
                K += 1
            # 将tmp加入结果
            res.append(tmp % 10)
        # A遍历完后，如果仍然K>0
        while K > 0:
            res.append(K % 10)
            K //= 10

        # 逆序输出结果
        return res[::-1]
```

#### 36.有效的数独
1. 一次迭代
	1. 算法思路
		1. 一个9X9的数独，要保证其有效性，需要达成三个条件：
			1. 行中没有重复数字
			2. 列中没有重复数字
			3. 3X3的子数独中没有重复数字
		2. 子数独block的下标**box\_idx=(行/3)\*3+列/3**，/为整除
		3. 遍历数独，**通过记录数字num和其出现的次数count在某行或某列或某子数独中的哈希映射关系**，当某行或某列或某子数独中的某个数字的count\>1时即出现了重复值。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，遍历了整个数独矩阵行M、列N
		2. 空间复杂度：O(N)，保存了行、列和子数独的哈希映射
	3. 题解代码
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 初始化
        row = [{} for _ in range(9)]
        col = [{} for _ in range(9)]
        box = [{} for _ in range(9)]

        # 遍历数独矩阵
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    # box下标
                    box_idx = (i // 3) * 3 + j // 3
                    # 记录num次数
                    row[i][num] = row[i].get(num, 0) + 1
                    col[j][num] = col[j].get(num, 0) + 1
                    box[box_idx][num] = box[box_idx].get(num, 0) + 1
                    # print('row[i][num]:' + str(row[i][num]) + ' col[j][num]:' + str(
                    #     col[j][num]) + ' box[box_idx][num]:' + str(box[box_idx][num]))
                    # 判断有效性
                    if row[i][num] > 1 or col[j][num] > 1 or box[box_idx][num] > 1:
                        return False
        return True
```

#### 1128.等价多米诺骨牌对的数量（本周复习）
1. 哈希表（字典）
	1. 算法思路
		1. 根据题目所示的**等价原则(a,b)=(b,a)**，即多米诺骨牌的a、b值只是顺序不同。
		2. 根据第1点，**可以有num=a\*10+b=b\*10+a，其中需要注意a和b的大小关系**。
		3. 建立一个字典，根据dict[num]=count次数来记录dominoes数组的情况，最终根据**组合公式n\*(n-1)/2**返回对数即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，字典保存记录所需的空间
	3. 题解代码
```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        from collections import defaultdict
        # 初始化
        dicts = defaultdict(int)
        # 遍历dominoes
        for i, j in dominoes:
            num = i * 10 + j if i < j else j * 10 + i
            dicts[num] += 1
        # 返回对数次数
        return sum(n * (n - 1) // 2 for n in dicts.values())
```

#### 485.最大连续1的个数
1. 数组一次遍历
	1. 算法思路
		1. 遍历一次数组nums，当nums[i]=1时，临时结果tmp+=1；否则最终结果res=max(res, tmp)，同时将tmp重置为0。
		2. 最终返回max(res,tmp)的较大者即可；此处为处理一些特殊情况，如nums=[1]。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组，长度为N
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        res = tmp = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                tmp += 1
            else:
                res = max(res, tmp)
                tmp = 0
        return max(res, tmp)
```

#### 51.N皇后
1. 回溯（递归法）
	1. 算法思路
		1. N皇后问题，是将N个皇后放在NXN的棋盘上，每个皇后之间不能互相攻击。因为皇后可以进行横向，纵向和斜向攻击，故意味着棋盘上**每一行有且仅有一个皇后，每一列有且仅有一个皇后，及每一条斜线有且仅有一个皇后**。满足这些条件时，即可得到一个可行解。
		2. 回溯+剪枝
			1. 回溯
				1. 建立NXN的棋盘，每个格子均为’.’，然后逐行将棋盘坐标(r,c)进行置放Q的尝试。
				2. 如果Q是无效的，则跳过；如果Q是有效的，则将(r,c)=Q，然后递归工作r+1行，递归工作后再将(r,c)置回为’.’进行回溯。递归工作中需要进行剪枝，优化时间复杂度。
			2. 剪枝
				1. 如果r行已有一个Q，其位置是(r,c)，那么r+1行递归工作时，进行剪枝判断。假设r+1当前的位置是(i,j)，且(i,j)=Q时，则有三种情况表示新Q与旧Q可以互相攻击：
					1. 当c=j时，表示两个Q位于同一列
					2. 当r+c=i+j时，表示两个Q位于45度角的同一斜线上。
					3. 当r-c=i-j时，表示两个Q位于135度角的同一斜线上。
				2. 符合以上三种情况进行表示(r,c)位置不可以放置Q，返回False，否则放回True。
	2. 复杂度分析
		1. 时间复杂度：O(N!)，N取决于皇后数量，因为遍历NXN的棋盘，所以是N！。
		2. 空间复杂度：O(N)，记录每个可能解的数组长度为N，系统调用递归的层数不超过N。
	3. 题解代码
```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 递归
        def dfs(r):
            if r == n:  # 递归终止条件
                res.append([''.join(row) for row in b])
                return
            else:
                for c in range(n):
                    if isValid(r, c):
                        b[r][c] = 'Q'
                        dfs(r + 1)
                        # 回溯
                        b[r][c] = '.'

        # 剪枝
        def isValid(r, c):
            for i in range(r):
                for j in range(n):  # 这里不是c，是n，表示每列都要检查
                    if b[i][j] == 'Q' and (c == j or r + c == i + j or r - c == i - j):
                        return False
            return True

        b = [['.'] * n for _ in range(n)]
        res = []
        dfs(0)  # 从第一行开始，第一行下标为0
        return res
```
4. 力扣国际站高赞题解代码（精简高效）
```python
def solveNQueens(self, n):
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p==n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])  
    result = []
    DFS([],[],[])
    return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]
```

#### 1143.最长公共子序列（本周复习）
1. 动态规划
	1. 算法思路
		1. 假设text1长度=m，text2长度=n，建立m\*n的矩阵dp，初始值=0。text1的每个字母对应**1\~m+1**的下标；同理可得，text2的每个字母对应**1\~n+1**的下标。**矩阵的[0,0]作为空头，方便书写状态转移方程**。
		2. 根据规律可得状态转移公式：
			1. 当text1[i-1]=text2[j-1]时，表示text1的字母等于text2的字母，这时子序列长度+1，即有dp[i][j]=dp[i-1][j-1]+1。
			2. 当text1[i-1]!=text2[j-1]时，表示text1的字母不等于text2的字母，这时dp[i][j]保持最大子序列的值，即有dp[i][j]=max(dp[i-1][j],dp[i][j-1])
		3. 最终返回dp[-1][-1]的值即可。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M为text1长度，N为text2长度。
		2. 空间复杂度：O(M\*N)，建立M\*N的矩阵空间，保存状态值。
	3. 题解代码
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 初始化dp矩阵
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # print(dp)
        # 遍历dp
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # print(dp)
        return dp[-1][-1]
```
4. dp状态压缩优化
	1. 将dp矩阵进行压缩，从二维数组变为一维数组。假设一维数组是text2的长度+1大小，dp=[0]\*(n+1)。
		2. 设置一个变量tmp用于转移dp[j]的状态值。
		3. 复杂度分析
			1. 时间复杂度：O(M\*N)
			2. 空间复杂度：O(N)，N为text2的长度
		4. 题解代码
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 初始化dp矩阵
        m, n = len(text1), len(text2)
        dp = [0] * (n + 1)
        # print(dp)
        # 遍历dp
        for i in range(1, m + 1):
            pre = 0  # text1的字母变化时，pre需置0
            for j in range(1, n + 1):
                tmp = dp[j]
                if text1[i - 1] == text2[j - 1]:
                    dp[j] = pre + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                pre = tmp
        # print(dp)
        return dp[-1]
```

#### 561.数组拆分I
1. 排序
	1. 算法思路
		1. 数组nums为2n，将其拆分成n对，取每一对的最小值并加总，返回最大加总值。
		2. 根据题例得出规律：将数组排序后，从小到大，小的数字组对，大的数字组对，然后取每对中的min值，加总值即最优解。
		3. 所以整个算法做两件事：
			1. 对数组nums升序排序
			2. for循环遍历数组，间隔为2，取i和i+1中较小者累加，返回累加值即可。
	2. 复杂度分析
		1. 时间复杂度：O(N\*logN)，遍历一次数组，加数组排序
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        for i in range(0, len(nums), 2):
            res += min(nums[i], nums[i + 1])
        return res
```
4. python切片优化
	1. 复杂度分析
		1. 时间复杂度：O(N\*logN)
			2. 空间复杂度：O(N)，系统切片占用空间
```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[::2])
```

#### 566.重塑矩阵
1. 原始数组一维化
	1. 算法思路
		1. 原始数组的行m=len(nums)，列n=len(nums[0])。**当待重塑的行列乘积不等于当前数组的行列乘积之时，即无法重塑，即m\*n!=r\*c时，返回原始nums；否则可以重塑矩阵**。
		2. 原始数组nums里的第x个元素为下标(i,j)，当nums压扁成一维数组后，**第x个元素在新数组中的下标为(i\*n+j)**。如果**将一维数组中第x个元素映射回原始矩阵，其下标i=x/n，j=x%n**。
		3. 同理，重塑矩阵res时，**第x个元素存在res[x/c][x%c]=nums[x/n][x%n]**的映射关系。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，可重塑时，为重塑矩阵的行列乘积
		2. 空间复杂度：O(M\*N)，重塑矩阵需要的空间。如果不含重塑矩阵的空间，即O(1)。
	3. 题解代码
```python
class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(nums), len(nums[0])  # 原始矩阵的行列
        if m * n != r * c:  # 如果新矩阵的行列乘积<>原始矩阵的行列乘积，即不能重塑
            return nums
        # 初始化重塑矩阵
        res = [[0] * c for _ in range(r)]
        for i in range(m * n):
            res[i // c][i % c] = nums[i // n][i % n]
        return res
```

2. 数组遍历
	1. 算法思路
		1. 与原始数据一维化处理同理，当原始数组的行列乘积不等于重塑数组的行列乘积之时，即无法重塑矩阵。
		2. 逐个遍历原始数组，将其元素复制到新矩阵中。设定新矩阵元素下标为(row,col)，当col=c时，即新矩阵该行填充完毕，需要换行处理，即row+1。
	2. 复杂度分析与原始数组一维化一致。
	3. 题解代码
```python
class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(nums), len(nums[0])  # 原始矩阵的行列
        if m * n != r * c:  # 如果新矩阵的行列乘积<>原始矩阵的行列乘积，即不能重塑
            return nums
        # 初始化重塑矩阵
        res = [[0] * c for _ in range(r)]
        row = col = 0
        for i in range(m):
            for j in range(n):
                if col == c:
                    row += 1
                    col = 0
                res[row][col] = nums[i][j]
                col += 1
        return res
```

#### 1306.跳跃游戏III
1. 广度优先搜索（BFS）
	1. 算法思路
		1. 建立队列queue，将起始下标start加入队列。
		2. 当queue不为空时进行循环搜索，具体为
			1. 取出队列的首端元素i，下一个新下标v=i+arr[i]或i-arr[i]。
			2. 当下标v在[0,len(arr))的范围内时，且v没有被搜索过，则加v加入队列queue，继续进行搜索。
			3. 当arr[v]=0时，即搜索完成，返回true；否则当队列为空后，仍没有符合条件的下标v，则返回false。
	2. 复杂度分析
		1. 时间复杂度：O(N)，N为数组长度
		2. 空间复杂度：O(N)，队列queue利用的空间
	3. 题解代码
```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        from collections import deque
        # 特例
        if arr[start] == 0:
            return True
        # 初始化
        queue = deque()
        queue.append(start)
        used = set()
        used.add(start)
        # 搜索队列
        while queue:
            i = queue.popleft()
            for j in [i + arr[i], i - arr[i]]:
                # 下标v在arr下标范围内，且未被使用过
                if 0 <= j < len(arr) and j not in used:
                    if arr[j] == 0:  # 找到目标返回true
                        return True
                    else:
                        queue.append(j)
                        used.add(j)
        # 搜索完毕后没有匹配的下标，则返回false
        return False
```

2. 深度优先搜索（DFS）
	1. 算法思路
		1. **建立一个used数组，记录arr数组每个下标idx是否被使用过**。如果递归工作时，发现下标idx已被使用，说明跳跃陷入循环，返回false。
		2. 如果下标idx未被使用过，且arr[idx]=0时，即找到目标，返回true。
		3. 不属于1、2点时，将下标idx标记为已使用；并判断idx+arr[idx]和idx-arr[idx]是否在[0,len(arr))的下标范围内，在范围内的下标继续递归工作，否则返回false；**这里注意向左向右跳的两种可能**。
		4. 最终返回dfs(start)的结果即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，最差遍历数组所有节点
		2. 空间复杂度：O(N)，递归时系统调用栈的空间
	3. 题解代码
```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        n = len(arr)
        used = [0] * n

        def dfs(pos):
            # 递归终止条件
            # pos下标已被使用
            if used[pos] == 1:
                return False
            # pos下标未被使用，且arr[pos]=0
            if used[pos] == 0 and arr[pos] == 0:
                return True
            # 标记pos已使用
            used[pos] = 1
            # 向左向右跳两种情况
            left = right = False
            if 0 <= pos + arr[pos] < n:
                left = dfs(pos + arr[pos])
            if 0 <= pos - arr[pos] < n:
                right = dfs(pos - arr[pos])
            return left or right

        return dfs(start)
```

#### 1.两数之和（本周复习）
1. 哈希字典
	1. 算法思路
		1. 创建一个字典hashmap；枚举数组nums，其中i为下标，v为值。
		2. 因为是在数组nums中寻找nums[i]+nums[j]=target，则有target-nums[i]=nums[j]。
		3. 在遍历数组nums的过程中，在字典hashmap中寻找target=v是否存在即可。如果hashmap[target-v]存在，返回hashmap.get(target-v)和下标i即可；否则将i和v存入字典中待用hashmap[v]=i。
	2. 复杂度分析
		1. 时间复杂度：O(N)，可能遍历整个数组的长度N。
		2. 空间复杂度：O(N)，字典所需的空间。
	3. 题解代码
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i, v in enumerate(nums):
            if hashmap.get(target - v) is not None:
                return [hashmap.get(target - v), i]
            else:
                hashmap[v] = i
```

#### 53.最大子序和（本周复习）
1. 贪心算法
	1. 算法思路
		1. 设定累加最大值max\_cur和当前值cur，初始都为nums[0]。
		2. 遍历数组nums（**从第2个元素开始遍历**）；先令cur为nums[i]与cur+nums[i]之间的较大者，即cur=max(cur,cur+nums[i])；再令max\_cur为max\_cur与cur之间的较大者，即max\_cur=max(max\_cur,cur)；最终返回max\_cur。
		3. 第2点思想体现在当数组遍历到第i个元素时，先看当前元素nums[i]与cur加上当前元素nums[i]后的和，它们之间谁大；再看max\_cur与第一步之后的当前cur之间谁大。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历整个数组。
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 贪心算法
        max_cur = cur = nums[0]
        for i in range(1, len(nums)):
            cur = max(nums[i], cur + nums[i])
            max_cur = max(max_cur, cur)
        return max_cur
```

2. 动态规划
	1. 算法思路
		1. 定义dp[i]为数组nums中已nums[i]为结尾的最大连续子序和，则有状态转移公式**dp[i]=max(dp[i-1]+nums[i],nums[i])**。
		2. 最终返回dp中的最大值即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)，额外数组dp保存了状态结果。
	3. 题解代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        # print(dp)
        return max(dp)
```

3. 分治法
	1. 算法思路
		1. 最大子序和可能存在于数组里的三个部分，分别是：
			1. 数组的左半部分
			2. 数组的右半部分
			3. 穿过数组中间，在数组中间部分。
		2. 左、右半部分的最大子序和可以调用递归计算
		3. 中间部分直接计算出来
			1. 从右往左计算左半部分的最大子序和
			2. 从左往右计算右半部分的最大子序和
			3. 两者相加即中间部分的最大子序和
		4. 最后返回左、中、右三个部分中的最大者，即数组的最大子序和。
	2. 复杂度分析
		1. 时间复杂度：O(NlogN)
		2. 空间复杂度：O(N)，递归时调用的系统栈
	3. 题解代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 分治
        n = len(nums)
        # 递归终止条件
        if n == 1:
            return nums[0]
        else:
            # 递归左半部分
            max_left = self.maxSubArray(nums[0:n // 2])
            # 递归右半部分
            max_right = self.maxSubArray(nums[n // 2:n])

        # 计数中间部分
        # 中间部分的左半部分
        max_l = nums[n // 2 - 1]
        tmp = 0
        for i in range(n // 2 - 1, -1, -1):
            tmp += nums[i]
            max_l = max(max_l, tmp)
        # 中间部分的右半部分
        max_r = nums[n // 2]
        tmp = 0
        for i in range(n // 2, n):
            tmp += nums[i]
            max_r = max(max_r, tmp)

        # 返回最终结果
        return max(max_left, max_right, max_l + max_r)
```

#### 15.三数之和（本周复习）
1. 双指针+排序
	1. 算法思路
		1. 先将数组nums排序
		2. 固定数组中最左数字的指针k，另外两个指针i，j分别为nums[k+1,len(nums)]的两端。通过i,j交替向中间移动，记录所有符合nums[k]+nums[i]+nums[j]=0的组合。
		3. 当nums[k]\>0时，break，因为nums[k]\>0，必然不会有nums[k]+nums[i]+nums[j]=0的组合。
		4. 当K\>0，且nums[k]==nums[k-1]时，跳过nums[k]。因为如果nums[k-1]的组合已经在结果里，那么nums[k]会造成重复结果。
		5. i,j分别为nums[k+1,len(nums)]的两端，当i\<j时循环，计算s=nums[k]+nums[i]+nums[j]，并根据以下情况判断：
			1. 当s\>0时，移动j-=1，并跳过所有重复的j，即nums[j]=nums[j+1]时continue。
			2. 当s\<0时，移动i+=1，并跳过所有重复的i，即nums[i]==nums[i-1]时continue。
			3. 当s=0时，符合目标结果，将[k,i,j]添加到结果res中，并i+=1,j-=1，并跳过所有重复的i,j。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)，指针k外循环O(N)，i、j内循环O(N)
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        n = len(nums)
        for k in range(n-2):
            if nums[k] > 0:  # 不会有可行的组合解
                break
            if k > 0 and nums[k] == nums[k - 1]:  # 跳过重复元素
                continue
            # 初始i,j
            i = k + 1
            j = n - 1
            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s < 0:  # s<0，移动i，跳过重复i
                    i += 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                elif s > 0:  # s>0，移动j，跳过重复j
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                else:  # s=0，记录k,i,j组合，移动i,j，跳过重复i,j
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
        return res
```

2. 递归法
	1. 算法思路
		1. 将问题拆解至求解两数之和的形式。
		2. 先定义一个递归函数求两数之和是否等于目标值。具体实现仍然是双指针法：
			1. 固定i,j为数组两端，当i\<j是循环，计算s=nums[i]+nums[j]的情况。
			2. 当s\<0时，移动i+=1；当s\>0时，移动j-=1；当s=0时，记录nums[i]和nums[j]，并移动i和j。
			3. 注意跳过重复元素。
		3. 对于三个数或以上，通过固定一个数，再将剩余的数进行递归拆解，直至转换成第2点的两数之和的问题。
	2. 复杂度分析
		1. 时机复杂度：O(N^n)，视需要拆解多少次，次数为n；每次递归工作时间复杂度为O(N)
		2. 空间复杂度：O(N)，递归工作时调用的系统栈
	3. 题解代码
```python
class Solution:
    # 定义两数之和的递归
    def dfs(self, nums, n, target):
        res = []
        if len(nums) < n:
            return res
        # n=2时
        if n == 2:
            i, j = 0, len(nums) - 1
            while i < j:
                s = nums[i] + nums[j]
                if s < target:
                    i += 1
                elif s > target:
                    j -= 1
                elif s == target:
                    res.append([nums[i], nums[j]])
                    while i < j and nums[i] == nums[i + 1]:
                        i += 1
                    while i < j and nums[j] == nums[j - 1]:
                        j -= 1
                    i += 1
                    j -= 1
                    
            return res
        else:  # 继续分解
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                subres = self.dfs(nums[i + 1:], n - 1, target - nums[i])
                # 将subres添加至res中
                for j in range(len(subres)):
                    res.append([nums[i]] + subres[j])
            return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        nums.sort()
        return self.dfs(nums, 3, 0)
```

#### 127.单词接龙（本周复习）
1. 单向BFS
	1. 算法思路
		1. 一开始判断下endWord是否在wordList中，不在可以直接返回0，因为不存在可能的解。
		2. **将wordList处理成set集合，有利于单词转换中对wordList的查询，降低时间复杂度**。（很重要）
		3. 创建一个队列queue，初始将beginWord和次数1加入，queue.append(beginWord,1)。
		4. 创建一个set集合visited，用于保存已访问过的单词，初始将beginWord加入。
		5. 当queue不为空时，循环遍历队列；取出队列首端元素赋给cur和count两个变量（cur,count=queue.popleft()），进行如下操作：
			1. 当cur等于endWord时，表示已匹配到目标单词，返回count。
			2. 当cur不等于endWord时，继续做以下操作：
				1. 遍历cur的每个字母，同时用26个字母逐个替换它，生成tmp单词。
				2. 当tmp单词在wordList中有匹配对象且不在visited里时，将tmp单词加入队列，次数count+1；同时将tmp单词放入visited集合，记录已被使用。
		6. 当队列为空后，如果没有匹配到目标单词，返回0，表示没有接龙成功。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M表示单词数量，N表示单词字母数。
		2. 空间复杂度：O(N)，N表示wordList和visited集合使用的空间。
	3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # wordlist——>set
        st = set(wordList)
        # endword不在st中，返回0
        if endWord not in st:
            return 0
        # 初始化队列queue和visited
        from collections import deque
        queue = deque()
        queue.append((beginWord, 1))
        visited = set()
        visited.add(beginWord)
        # 遍历队列
        while queue:
            cur, count = queue.popleft()
            # 找到目标单词，返回次数
            if cur == endWord:
                return count
            # 没找到目标，继续处理
            for i in range(len(cur)):
                # 单词字母替换
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    tmp = cur[:i] + j + cur[i + 1:]
                    # tmp未被使用过，且在wordlist中
                    if tmp in st and tmp not in visited:
                        queue.append((tmp, count + 1))
                        visited.add(tmp)
        # 没有可能解，返回0
        return 0
```

2. 双向BFS
	1. 算法思路
		1. 在单向BFS上进行优化处理。与单向BFS不同之处在于，对beginWord和endWord分别做了队列和记录使用的集合，lqueue、lvisited、rqueue和rvisited。
		2. **以层为单位记录接龙次数**。只有当队列里当前元素都处理完了，才对接龙次数加一，这点与单向BFS不同。
		3. 当lqueue和rqueue均不为空时，先判断哪个队列较短，后续对较短的队列进行操作即可。假设lqueue队列为元素少的队列：
			1. 从lqueue中取出首端元素cur，当cur在rvisited中已经存在，即说明接龙成功，返回层数。
			2. 当cur不在rvisited中时，仍然用26字母逐个替换cur单词的字母，生成tmp。
			3. 当tmp不在lvisited中，又在wordList中时，将tmp加入lvisited和lqueue中。
		4. 如最终没有接龙成功，返回0。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M为单词数，N为单词字母个数。
		2. 空间复杂度：O(N)，队列和集合需要利用额外空间。
	3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # wordlist——>set
        st = set(wordList)
        # endword不在st中，返回0
        if endWord not in st:
            return 0
        # 初始化队列queue和visited
        from collections import deque
        lqueue = deque()
        lqueue.append(beginWord)
        lvisited = set()
        lvisited.add(beginWord)
        rqueue = deque()
        rqueue.append(endWord)
        rvisited = set()
        rvisited.add(endWord)
        step = 0
        # 遍历队列
        while lqueue and rqueue:
            # 找出元素少的队列
            if len(lqueue) > len(rqueue):
                lqueue, lvisited, rqueue, rvisited = rqueue, rvisited, lqueue, lvisited
            # 次数+1
            step += 1
            # 对短队列处理每一个元素
            for k in range(len(lqueue)):
                cur = lqueue.popleft()
                # 在rvisited找到目标单词，返回次数
                if cur in rvisited:
                    return step
                # 没找到目标，继续处理
                else:
                    for i in range(len(cur)):
                        # 单词字母替换
                        for j in 'abcdefghijklmnopqrstuvwxyz':
                            tmp = cur[:i] + j + cur[i + 1:]
                            # tmp未被使用过，且在wordlist中
                            if tmp in st and tmp not in lvisited:
                                lqueue.append(tmp)
                                lvisited.add(tmp)
        # 没有可能解，返回0
        return 0
```

#### 18.四数之和（本周复习）
1. 递归法
	1. 算法思路
		1. **将问题拆解至求解两数之和的形式**。
		2. 先定义一个递归函数求两数之和是否等于目标值。具体实现仍然是双指针法：
			1. 固定i,j为数组两端，当i\<j是循环，计算s=nums[i]+nums[j]的情况。
			2. 当s\<0时，移动i+=1；当s\>0时，移动j-=1；当s=0时，记录nums[i]和nums[j]，并移动i和j。
			3. 注意跳过重复元素。
		3. 对于三个数或以上，通过固定一个数，再将剩余的数进行递归拆解，直至转换成第2点的两数之和的问题。
		4. 对于四数之和或N数之和，不断重复1、2、3点进行递归工作即可。
	2. 复杂度分析
		1. 时机复杂度：O(N^n)，视需要拆解多少次，次数为n；每次递归工作时间复杂度为O(N)
		2. 空间复杂度：O(N)，递归工作时调用的系统栈
	3. 题解代码
```python
class Solution:
    # 递归nSum
    def nSum(self, nums, n, target):
        res = []
        if len(nums) < n:
            return res
        # 当n==2时
        if n == 2:
            i, j = 0, len(nums) - 1
            while i < j:
                s = nums[i] + nums[j]
                if s == target:
                    res.append([nums[i], nums[j]])
                    # 跳过重复元素
                    while i < j and nums[i] == nums[i + 1]:
                        i += 1
                    while i < j and nums[j] == nums[j - 1]:
                        j -= 1
                    i += 1
                    j -= 1
                elif s < target:
                    i += 1
                else:
                    j -= 1
            return res
        else:  # n!=2时
            for k in range(len(nums)):
                if k > 0 and nums[k] == nums[k - 1]:
                    continue
                else:
                    subres = self.nSum(nums[k + 1:], n - 1, target - nums[k])
                    for j in range(len(subres)):
                        res.append([nums[k]] + subres[j])
            return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        return self.nSum(nums, 4, target)
```

#### 697.数组的度
1. 哈希表
	1. 算法思路
		1. 数组的度是数组中各元素出现的最大值。
		2. 与原数组度相同的连续子数组的头尾两个元素，必然是出现次数最多元素的第一次和最后一次出现的位置。在所有符合的子数组中找出最短的那个即可。
		3. 具体实现：
			1. 通过哈希表字典dict保存数组中元素的出现次数，第一次出现的下标和最后一次出现的下标。
			2. 设定maxnum和minlen两个变量，一个表示数组的度，一个表示最短子数组长度。
			3. 遍历dict，设count, left, right分别为数组的度，元素第一次位置和最后一次位置。
				1. 当maxnum\<count时，maxnum=count，同时minlen=right-left+1。
				2. 当maxnum==count时，minlen=min(minlen,right-left+1)。
				3. 返回最终的minlen即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组+遍历一次字典，O(N)+O(N)
		2. 空间复杂度：O(N)，字典利用额外空间
	3. 题解代码
```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        # 初始化哈希字典
        dict = {}
        # 记录相同元素出现的次数，第一次位置和最后一次位置
        for i, v in enumerate(nums):
            if v in dict:
                dict[v][0] += 1
                dict[v][2] = i
            else:
                dict[v] = [1, i, i]
        # print(dict)
        # print(dict.values())
        # print(dict.items())
        # 设定度，最短长度
        maxcount = minlen = 0
        for count, left, right in dict.values():
            if maxcount < count:  # 度不是最大时，不断更新度和最短长度
                maxcount = count
                minlen = right - left + 1
            elif maxcount == count:  # 度一样时找最短长度
                minlen = min(minlen, right - left + 1)
        return minlen
```

#### 21.合并两个有序链表（本周复习）
1. 递归法
	1. 算法思路
		1. 当l1或l2为空时，递归终止。
		2. 根据l1和l2的头节点大小，让较小者的next指针指向其余已合并的所有节点。
		3. 其余已合并的所有节点即是通过递归调用生成的。
	2. 复杂度分析
		1. 时间复杂度：O(M+N)，M为l1长度，N为l2长度
		2. 空间复杂度：O(M+N)，递归时的系统栈，栈空间根据两个链表的长度决定。
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 终止条件
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        # 较小节点的next，指向其余节点的合并结果
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

#### 22.括号生成（本周复习）
1. 递归法
	1. 算法思路
		1. 左右括号的个数都为n，生成n对左右括号对。通过递归工作，不断的在一个初始为空的字符串中添加左括号和右括号来达成目的。
		2. **本题的难点在于生成括号对的过程中去除无效的括号对，只保留有效括号对**。
		3. 递归工作过程：
			1. 当**左括号个数left和右括号个数right都等于n时，表示已经添加了n对的括号对**，返回字符串结果即可。
			2. 为了只保留有效括号对，所以在生成括号的过程中进行**剪枝处理**，具体剪枝原理为：
				1. 当左括号个数left\<n时，可以添加左括号，同时左括号个数left+1，dfs(cur\_str+’(‘,left+1,right)。
				2. 当已有左括号，且左括号个数left\>右括号个数right时，可以添加右括号，同时右括号个数right+1，dfs(cur\_str+’)’,left,right+1)。
				3. 第2点中有个间接关系，即右括号个数right也要满足\<n，但因为第1点已满足左括号个数left\<n，所以这个间接条件可以写在判断表达式中，也可以不写。
	2. 复杂度分析
		1. 时间复杂度：O(N)，左右括号个数均为n，实际时O(2N)=O(N)
		2. 空间复杂度：O(N)，递归时系统栈的使用次数等于括号个数n。
	3. 题解代码
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # 递归法
        def dfs(cur_str, left, right):
            if left == n and right == n:
                res.append(cur_str)
                return
                # 递归及剪枝
            if left < n:
                dfs(cur_str + '(', left + 1, right)
            if left > right:
                dfs(cur_str + ')', left, right + 1)

        # 初始化及返回结果
        res = []
        dfs('', 0, 0)
        return res
```


[1]:	https://leetcode-cn.com/problems/implement-trie-prefix-tree/
[2]:	https://leetcode-cn.com/problems/number-of-provinces/
[3]:	https://www.cnblogs.com/fusiwei/p/11759489.html
[4]:	https://leetcode-cn.com/problems/climbing-stairs/
[5]:	https://leetcode-cn.com/problems/generate-parentheses/
[6]:	https://leetcode-cn.com/problems/n-queens/
[7]:	https://leetcode-cn.com/problems/word-ladder/
[8]:	https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/
[9]:	https://leetcode-cn.com/problems/sliding-puzzle/
[10]:	https://www.cnblogs.com/ISGuXing/p/9800490.html
[11]:	https://dataaspirant.com/five-most-popular-similarity-measures-implementation-in-python/
[12]:	https://zxi.mytechroad.com/blog/searching/8-puzzles-bidirectional-astar-vs-bidirectional-bfs/

