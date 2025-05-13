分四步：
第一步：目标为偏离，无论偏离到哪个产品
第二步：目标为偏离到我们指定的产品
第三步：joint-optimization GCG，compression model和后面的LLM都是白盒
第四步：transfer attack，即我们根据1，2，3步得到了一些能够影响embedding的token，我们把这些token加到别的产品demo上去，看看会不会有影响，相当于attack的generalization。
一步一步来，不着急。
最后，我会搭一个实际的agent system，我们在real-world agent上测。