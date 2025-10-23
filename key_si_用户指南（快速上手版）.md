# KeySIUserInstruction

This document provides a simple guide to using KeySI.

---

### Before You Start
Make sure all required packages listed in the **requirements.txt** file are installed. You can also change the output directory in **KEYSI_UI.py** by modifying the path variable `OUTPUT_DIR = "KeySI_results"` if you want to save results to another folder.

---

### Main Steps
1. **Select Group Number**  
   In the top left corner, choose the number of groups you want to generate.

2. **Generate Groups**  
   Click the **Generate** button on the left. New groups such as *Group1, Group2, ...* will appear in the lower-left panel.

3. **Assign Keywords**  
   Click a group name (for example *Group1*). Then, in the top-left 2D keyword plot, select the keywords you want to add to that group.

4. **Exclude Irrelevant Items**  
   If some documents are not useful, you can put them into the **Exclude** group to remove them from training.

5. **Start Training**  
   Click the **Train** button. It will show *Training...* on the screen. When training finishes, a new page will appear called **Training View**.

---

### Training View
In **Training View**, you can see the document positions before and after training. This allows you to compare how the model changes the document distribution. If you are not satisfied, you can go back and adjust group assignments or keywords and train again.

---

### FindTurningView
After training, you can open **FindTurningView** to inspect documents used in model training. Here you can check each document’s position and decide whether to move it between groups (for example, from *Group1* to *Group2*) or move it to the **Exclude** group if it doesn’t belong anywhere. You can also click points in the 2D visualization to check their similarity and decide if they should remain in their group or be reassigned. After finishing adjustments, you can click **FindTurning** again to generate a new view. You can repeat this process as many times as needed.

---

That’s all you need to know to get started with KeySI.

