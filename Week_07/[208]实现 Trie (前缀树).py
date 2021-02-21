# 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。 
# 
#  示例: 
# 
#  Trie trie = new Trie();
# 
# trie.insert("apple");
# trie.search("apple");   // 返回 true
# trie.search("app");     // 返回 false
# trie.startsWith("app"); // 返回 true
# trie.insert("app");   
# trie.search("app");     // 返回 true 
# 
#  说明: 
# 
#  
#  你可以假设所有的输入都是由小写字母 a-z 构成的。 
#  保证所有输入均为非空字符串。 
#  
#  Related Topics 设计 字典树 
#  👍 518 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
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
# leetcode submit region end(Prohibit modification and deletion)
