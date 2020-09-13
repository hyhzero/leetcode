## 简单

### 1. 数组中重复的数字

* 计数

  ```python
  class Solution:
      def findRepeatNumber(self, nums: List[int]) -> int:
  
          counts = {}
          for num in nums:
              if num in counts:
                  return num
              else:
                  counts[num] = 1
  ```

* 集合

  ```python
  class Solution:
      def findRepeatNumber(self, nums: List[int]) -> int:
  
          ans = set()
          for num in nums:
              if num in ans:
                  return num
              else:
                  ans.add(num)
  ```

* 桶排序

  ```python
  class Solution:
      def findRepeatNumber(self, nums: List[int]) -> int:
  
          i = 0
          while i < len(nums):
              if nums[i]==i:
                  i += 1
              elif nums[i]==nums[nums[i]]:
                  return nums[i]
              else:
                  temp = nums[i]
                  nums[i] = nums[temp]
                  nums[temp] = temp
  ```

### 2. 两个链表的第一个公共节点

* 消除距离差

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
  
          p = headA
          q = headB
          while p!=q:
  
              p = p.next if p else headB
              q = q.next if q else headA
          
          return p
  ```

### 3. 在排序数组中查找数字

* 扫描

  ```python
  class Solution:
      def search(self, nums: List[int], target: int) -> int:
  
          i = 0
          while i < len(nums) and target!=nums[i]:
              i += 1
  
          if i >= len(nums):
              return 0
  
          ans = 0
          while i < len(nums) and target==nums[i]:
              ans += 1
              i += 1
  
          return ans
  ```

### 4. 二维数组中的查找

* 暴力

  ```python
  class Solution:
  
      def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
          i = 0
          j = 0
          while i < len(matrix) and j < len(matrix[0]) and matrix[i][j] <= target:
              k = j
              while k < len(matrix[0]) and matrix[i][k] < target:
                  k += 1
              
              if k < len(matrix[0]) and matrix[i][k]==target:
                  return True
              
              k = i
              while k < len(matrix) and matrix[k][j] < target:
                  k += 1
              
              if k < len(matrix) and matrix[k][j]==target:
                  return True
              
              i += 1
              j += 1
         
          return False
              
  ```

* 右上角查找

  ```python
  class Solution:
  
      def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
          if len(matrix)==0:
              return False
          
          i = 0
          j = len(matrix[0]) - 1
  
          while i >= 0 and i < len(matrix) and j >= 0 and j < len(matrix[0]):
              
              if target==matrix[i][j]:
                  return True
  
              if target > matrix[i][j]:
                  i += 1
              else:
                  j -= 1
  
          return False
  ```





### 5. 0～n-1中缺失的数字

* 作差

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          n = len(nums)
          sums = n*(n+1)//2
          return sums - sum(nums)
  ```

* 异或

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          ans = len(nums)
          for i, num in enumerate(nums):
              ans ^= i ^ num
          
          return ans
  ```

* 桶排序

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          i = 0
          while i < len(nums):
              if i==nums[i] or nums[i]==len(nums):
                  i += 1
              else:
                  temp = nums[i]
                  nums[i] = nums[temp]
                  nums[temp] = temp
  
          for i, num in enumerate(nums):
              if i!=num:
                  return i
              
          return len(nums)
  ```


### 6. 替换空格

* 暴力

  ```python
  class Solution:
      def replaceSpace(self, s: str) -> str:
  
          s = list(s)
          ans = []
          for x in s:
              if x==' ':
                  ans.append('%20')
              else:
                  ans.append(x)
  
          return ''.join(ans)
  ```

### 7. 从尾到头打印链表

* 逆序

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def reversePrint(self, head: ListNode) -> List[int]:
          
          ans = []
          p = head
          while p:
              ans.append(p.val)
              p = p.next
          
          ans = list(reversed(ans))
  
          return ans
  ```

* 递归

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
  
      def reverse(self, head, ans):
          if head==None:
              return
          
          if head.next==None:
              ans.append(head.val)
          else:
              self.reverse(head.next, ans)
              ans.append(head.val)
  
      def reversePrint(self, head: ListNode) -> List[int]:
          
          ans = []
          self.reverse(head, ans)
  
          return ans
  ```

### 8. 斐波那契数列

* 暴力

  ```python
  class Solution:
      def fib(self, n: int) -> int:
  
          if n <= 1:
              return n
  
          f0 = 0
          f1 = 1
  
          for _ in range(1, n):
              temp = f1
              f1 += f0
              f0 = temp
  
          return f1%1000000007
  ```

### 9. 青蛙跳台阶

* 动态规划

  ```python
  class Solution:
      def numWays(self, n: int) -> int:
          if n==0:
              return 1
          if n==1 or n==2:
              return n
  
          opt = [0 for _ in range(n)]
          opt[0] = 1
          opt[1] = 2
          for i in range(2, n):
              opt[i] = opt[i-1] + opt[i-2]
  
          return opt[-1]%1000000007
  ```

* 动态规划2

  ```python
  class Solution:
      def numWays(self, n: int) -> int:
          if n==0:
              return 1
          if n==1 or n==2:
              return n
  
          opt1 = 1
          opt2 = 2
  
          for _ in range(2, n):
              temp = opt2
              opt2 += opt1
              opt1 = temp
          
          return opt2%1000000007
  ```

### 10. 旋转数组的最小元素

* 从尾扫描

  ```python
  class Solution:
      def minArray(self, numbers: List[int]) -> int:
  
          i = len(numbers)-1
          while i > 0 and numbers[i] >= numbers[i-1]:
              i -= 1
          
          return numbers[i]
  ```


### 11. 二进制中1的个数

* 暴力

  ```python
  class Solution:
      def hammingWeight(self, n: int) -> int:
  
          ans = 0
          while n:
              ans += n%2
              n = n//2
          
          return ans
  ```

### 12. 打印从1到n的最大位数

* 暴力

  ```python
  class Solution:
      def printNumbers(self, n: int) -> List[int]:
  
          end = 0
          for _ in range(n):
              end = end*10 + 9
  
          return [i for i in range(1, end+1)]
  ```

### 13. 和为s的两个数字

* 集合

  ```python
  class Solution:
      def twoSum(self, nums: List[int], target: int) -> List[int]:
  
          sums = set()
          for x in nums:
              if target-x in sums:
                  return [x, target-x]
              else:
                  sums.add(x)
  ```

* 双指针

  ```python
  class Solution:
      def twoSum(self, nums: List[int], target: int) -> List[int]:
  
          i = 0
          j = len(nums)-1
          while i < j:
              if nums[i] + nums[j] == target:
                  return [nums[i], nums[j]]
              elif nums[i] + nums[j] > target:
                  j -= 1
              else:
                  i += 1
  ```

### 14. 和为s的连续正数序列

* 等差数列

  ```python
  class Solution:
      def findContinuousSequence(self, target: int) -> List[List[int]]:
  
          a = 1
          b = 2
          ans = []
          while a + b <= target:
              if (target-a)%b==0:
                  f = (target-a)//b
                  ans.append([f+i for i in range(b)])
  
              temp = b
              a += b
              b = temp + 1
  
          return list(reversed(ans))
  ```

### 15. 最小的K个数

* 排序

  ```python
  class Solution:
      def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
  
          arr.sort()
          return arr[:k]
  ```

* 堆排序

  ```python
  class Solution:
  
      def heapify(self, arr, i, length):
          left = 2*i + 1
          right = 2*i + 2
  
          max_i = i
  
          if left < length and arr[left] > arr[max_i]:
              max_i = left
          
          if right < length and arr[right] > arr[max_i]:
              max_i = right
          
          if i != max_i:
              temp = arr[i]
              arr[i] = arr[max_i]
              arr[max_i] = temp
              self.heapify(arr, max_i, length)
  
  
      def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
  
          i = len(arr)//2 - 1
          while i >= 0:
              self.heapify(arr, i, len(arr))
              i -= 1
  
          i = len(arr)-1
          while i >= 0:
              temp = arr[i]
              arr[i] = arr[0]
              arr[0] = temp
              self.heapify(arr, 0, i)
              i -= 1
          
          return arr[:k]
  ```

* 部分堆排序

  ```python
  class Solution:
  
      def heapify(self, arr, i, length):
          left = 2*i + 1
          right = 2*i + 2
  
          min_i = i
  
          if left < length and arr[left] < arr[min_i]:
              min_i = left
          
          if right < length and arr[right] < arr[min_i]:
              min_i = right
          
          if i != min_i:
              temp = arr[i]
              arr[i] = arr[min_i]
              arr[min_i] = temp
              self.heapify(arr, min_i, length)
  
  
      def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
  
          i = len(arr)//2 - 1
          while i >= 0:
              self.heapify(arr, i, len(arr))
              i -= 1
  
          i = len(arr)-1
          while i >= len(arr)-k:
              temp = arr[i]
              arr[i] = arr[0]
              arr[0] = temp
              self.heapify(arr, 0, i)
              i -= 1
          
          return arr[:-(k+1):-1]
  ```


### 16. 删除链表的节点

* 线性扫描

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def deleteNode(self, head: ListNode, val: int) -> ListNode:
  
          if head.val==val:
              return head.next
  
          p = head
          while p.next and p.next.val!=val:
              p = p.next
          
          if p.next:
              q = p.next
              p.next = q.next
  
          return head
  ```

### 17. 调整数组顺序使奇数位于偶数前面

* 双指针

  ```python
  class Solution:
      def exchange(self, nums: List[int]) -> List[int]:
  
          i = 0
          j = len(nums)-1
          
          while i < j:
              
              while i < j and nums[i]%2==1:
                  i += 1
              
              while i < j and nums[j]%2==0:
                  j -= 1
  
              if i < j:
                  temp = nums[i]
                  nums[i] = nums[j]
                  nums[j] = temp
  
          return nums
  ```

### 18. 链表中倒数第k个节点

* 两次循环

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
  
          p = head
          count = 0
          while p:
              count += 1
              p = p.next
          
          p = head
          for _ in range(count-k):
              p = p.next
  
          return p
  ```

* 快慢指针

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
  
          p = head
          
          for _ in range(k-1):
              p = p.next
  
          q = p
          p = head
          while q.next:
              q = q.next
              p = p.next
          
          return p
  ```

### 19. 反转链表

* 线性扫描

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def reverseList(self, head: ListNode) -> ListNode:
  
          if head==None or head.next==None:
              return head
  
          q = head
          p = head.next
          head.next = None
  
          while p:
              temp = p.next
              p.next = q
              q = p
              p = temp
  
          return q
  
  ```

* 栈

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
  
  
      def reverseList(self, head: ListNode) -> ListNode:
          if head==None or head.next==None:
              return head
  
          stack = []
          p = head
          while p:
              stack.append(p)
              p = p.next
  
          head = stack[-1]
          while len(stack) > 0:
              top = stack.pop()
              if len(stack) > 0:
                  top.next = stack[-1]
              else:
                  top.next = None
  
          return head
  ```

### 20. 合并两个排序的链表

* 线性扫描

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
  
          head = ListNode(0)
          p = head
          while l1 and l2:
              if l1.val < l2.val:
                  p.next = l1
                  l1 = l1.next
              else:
                  p.next = l2
                  l2 = l2.next
              p = p.next
  
          while l1:
              p.next = l1
              l1 = l1.next
              p = p.next
          while l2:
              p.next = l2
              l2 = l2.next
              p = p.next
  
          return head.next
  ```

* 递归

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
          if l1==None:
              return l2
          if l2==None:
              return l1
          
          if l1.val < l2.val:
              l1.next = self.mergeTwoLists(l1.next, l2)
              return l1
          else:
              l2.next = self.mergeTwoLists(l1, l2.next)
              return l2
  
  ```

### 21. 二叉树的镜像

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def mirrorTree(self, root: TreeNode) -> TreeNode:
  
          if root==None:
              return None
  
          left = self.mirrorTree(root.left)
          right = self.mirrorTree(root.right)
  
          root.left = right
          root.right = left
  
          return root
  ```

* 深度优先搜索

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def mirrorTree(self, root: TreeNode) -> TreeNode:
  
          stack = []
          stack.append(root)
  
          while len(stack) > 0:
              top = stack.pop()
              if top:
                  left = top.left
                  right = top.right
                  top.left = right
                  top.right = left
                  if top.right:
                      stack.append(top.right)
                  if top.left:
                      stack.append(top.left)
  
          return root
  ```

### 22. 对称二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def symmetric(self, left, right):
          if left==None and right==None:
              return True
  
          if left==None and right or left and right==None:
              return False
          
          if left.val==right.val:
              return self.symmetric(left.right, right.left) and self.symmetric(left.left, right.right)
          
  
      def isSymmetric(self, root: TreeNode) -> bool:
          if root==None:
              return True
          
          return self.symmetric(root.left, root.right)
  ```


### 23. 左旋转字符串

* 暴力

  ```python
  class Solution:
      def reverseLeftWords(self, s: str, n: int) -> str:
  
          k = n%len(s)
  
          s = list(s)
  
          ans = s[k:] + s[:k]
          return ''.join(ans)
  ```

### 24. 连续子数组最大和

* 动态规划？

  ```python
  class Solution:
      def maxSubArray(self, nums: List[int]) -> int:
  
          sums = 0
          ans = nums[0]
          for num in nums:
              if sums > 0:
                  sums += num
              else:
                  sums = num
              ans = max(ans, sums)
  
          return ans
  ```

* 动态规划

  ```python
  class Solution:
      def maxSubArray(self, nums: List[int]) -> int:
  
          if len(nums)==1:
              return nums[0]
  
          opt = [0 for _ in range(len(nums))]
          opt[0] = nums[0]
          for i in range(1, len(nums)):
              if opt[i-1] < 0:
                  opt[i] = nums[i]
              else:
                  opt[i] = nums[i] + opt[i-1]
  
          return max(opt)
  ```

* 分治法

  ```python
  class Solution:
  
      def maxsub(self, nums, left, right):
          if left==right:
              return nums[left]
  
          mid = (left + right)//2
          lm = self.maxsub(nums, left, mid)
          rm = self.maxsub(nums, mid+1, right)
          print(lm, rm, mid+1)
  
          i = mid
          max_v1 = nums[i]
          sums = 0
          while i >= left:
              sums += nums[i]
              max_v1 = max(max_v1, sums)
              i -= 1
  
          i = mid + 1
          max_v2 = nums[i]
          sums = 0
          while i <= right:
              sums += nums[i]
              max_v2 = max(max_v2, sums)
              i += 1
  
          max_v = max_v1 + max_v2
          return max([max_v, lm, rm])
  
  
  
      def maxSubArray(self, nums: List[int]) -> int:
  
          return self.maxsub(nums, 0, len(nums)-1)
  ```

  

### 25. 滑动窗口的最大值

* 暴力法

  ```python
  class Solution:
      def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
          if len(nums)==0:
              return []
  
          ans = []
          i = 0
          while (i+k) <= len(nums):
              ans.append(max(nums[i:i+k]))
              i += 1
          
          return ans
  ```

### 26. 二叉树的深度

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def maxDepth(self, root: TreeNode) -> int:
  
          if root==None:
              return 0
  
          return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
  ```

* 递归2

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def maxDepth(self, root: TreeNode) -> int:
  
          if root==None:
              return 0
  
          return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
  ```

* 先序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
  
      def maxDepth(self, root: TreeNode) -> int:
          
          if root==None:
              return 0
  
          ans = 0
          stack = []
          stack.append((root, 1))
          while len(stack) > 0:
              node, d = stack.pop()
              ans = max(ans, d)
              if node.left:
                  stack.append((node.left, d+1))
              if node.right:
                  stack.append((node.right, d+1))
              
          return ans
  ```

### 27. 扑克牌中的顺子

* 暴力

  ```python
  class Solution:
      def isStraight(self, nums: List[int]) -> bool:
  
          nums.sort()
  
          i = 0
          while i < len(nums) and nums[i]==0:
              i += 1
              
          count = i
  
          sums = 0
          while i < len(nums)-1:
              if nums[i]==nums[i+1]:
                  return False
              sums += nums[i+1] - nums[i] - 1
              i += 1
  
          return sums <= count
  ```

### 28. 平衡二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def depth(self, root):
          if root==None:
              return 0
          
          return max(self.depth(root.left), self.depth(root.right)) + 1
  
      def isBalanced(self, root: TreeNode) -> bool:
  
          if root==None:
              return True
  
          if abs(self.depth(root.left)-self.depth(root.right)) > 1:
              return False
  
          return self.isBalanced(root.left) and self.isBalanced(root.right)
  ```

### 29. 翻转单词顺序

* 暴力

  ```python
  class Solution:
      def reverseWords(self, s: str) -> str:
  
          i = len(s)-1
          ans = []
          while i >= 0:
  
              while i >= 0 and s[i]==' ':
                  i -= 1
              
              if i < 0:
                  break
  
              j = i
              while j >= 0 and s[j]!=' ':
                  j -= 1
  
  
              ans.append(s[j+1:i+1])
  
              i = j
  
          return ' '.join(ans)
  ```

### 30. 第一个只出现一次的字符

* 计数法

  ```python
  class Solution:
      def firstUniqChar(self, s: str) -> str:
  
          counts = {}
          for x in s:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          for k, v in counts.items():
              if v==1:
                  return k
  
          return " "
  ```

  