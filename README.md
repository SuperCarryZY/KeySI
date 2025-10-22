Define Keyword Groups
In the left-side Keyword Management Panel, click “+ Add Group”.


Enter a group name (e.g., sports, computer, politics).


Add 3–10 representative keywords for each group.


Example(You can easily copy past keywords with comma and enter the keyword):
sports: nhl,hockey,espn  
computer: software,modem,hardware

(Recommended/Optional) Add an Exclude Group to filter out unrelated documents.


Example:

 Exclude: war,crime, kill,tax


This group is not used for training; it serves only as a negative sample pool to help the model separate irrelevant content.




Train the Model
Click “Train Model” to start the initial training.


The system will run for about 10 epochs.


Once training completes,


The left panel shows the original projection,


The right panel updates with the new semantic projection in real time.



Fine-tune the Model
Click “Finetune Mode” to enter the interactive adjustment phase.


Select a group to review.


Focus on documents marked with gray highlights — these are uncertain or boundary samples.


Since each document now has a label, you can also directly check its assigned label on the right panel.


Reassign any documents that do not belong to the current group to another group or to the Other group.


Click “Finetune” to update the model.


This step allows the model to align with your latest semantic understanding, refining group boundaries based on your manual adjustments.


