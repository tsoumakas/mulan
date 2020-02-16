package mulan.sampling;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import mulan.data.*;
import weka.core.Instance;
import weka.core.Instances;

/***
* <p>Implementation of the MLRUS.</p> <p>For
* more information, see <em> Charte, Francisco, et al. 
* "Addressing imbalance in multilabel classification: Measures and random resampling algorithms."
*  Neurocomputing 163 (2015): 3¨C16.</em></p>
*
* @author Bin Liu
 */


public class MutilLabelRandomUnderSampling extends MultiLabelSampling{

	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception{
		Random rnd=new Random(seed);
		
		MultiLabelInstances newMlDataset=null;
		Instances ins=mlDataset.getDataSet();
		
		int oriInsNum=mlDataset.getNumInstances();
		int deleteInsNum=(int)(P*oriInsNum);
		int labelNum=mlDataset.getNumLabels();
		int labelIndices[]=mlDataset.getLabelIndices();
		boolean isMajLabel[]=new boolean[labelNum];
		HashSet<Integer> deleteInsIndexSet=new HashSet<Integer>();
		
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateImSta(mlDataset);
		
		double IRLbls[]=is.getIRLbls();
		double MeanIR=is.getMeanIR();
		int c1[]=is.getC1();
		
		for(int i=0;i<labelNum;i++){
			if(IRLbls[i]<MeanIR){
				isMajLabel[i]=true;
			}
		}
		
		ArrayList<ArrayList<Integer>> majBags=new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<labelNum;i++)
		{
			majBags.add(new ArrayList<Integer>());
		}
		
		
		
		for(int i=0;i<ins.numInstances();i++){
			Instance d=ins.get(i);
			for(int j=0;j<labelNum;j++){
				/*for d.value(j) the specified value as a double 
				(If the corresponding attribute is nominal (or a string) then it returns the value's index as a double).
				*/
				//d.attribute(labelIndices[j]).value((int)d.value(labelIndices[j]));
				if(d.stringValue(labelIndices[j]).equals("1")) 
				{
					if(isMajLabel[j])
						majBags.get(j).add(i);
				}
			}
		}
		int mc=Integer.MIN_VALUE;
		for(int i:c1){
			if(mc<i)
				mc=i;
		}
		
		boolean isAllMinority;
		while(deleteInsNum>0){
			
			isAllMinority=true;
			for(boolean b:isMajLabel){
				if(b){
					isAllMinority=false;
				}
			}
			if(isAllMinority){
				break;  
				/*
				 *  when all labels is minority (>= MeanIR) there is another solution that recalculating MeanIR, IRLbl, c1[] and mc
				 */
			}
			
			
			//Iterator <ArrayList<Integer>>it= majBags.iterator();
	        for(int i=0; i<labelNum; i++){
	        	if(!isMajLabel[i])
	        		continue;  //not a majority label
	        	
	        	//recalculate IRLbl of label i
	        	double newIRLbl= mc*1.0/c1[i]; 
	            if(newIRLbl>=MeanIR){
	            	isMajLabel[i]=false;  //tag label i to the minority label
	            	continue;
	            }
	               
	            int k=ImUtil.randomInt(0, majBags.get(i).size()-1,rnd);
	            int insIndex=majBags.get(i).get(k);
            	while(majBags.get(i).size()>0 && deleteInsIndexSet.contains(insIndex))
            	{
            		majBags.get(i).remove(k);
            		k=ImUtil.randomInt(0, majBags.get(i).size()-1,rnd);
	                insIndex=majBags.get(i).get(k);
            	} 
	            if(deleteInsIndexSet.contains(insIndex)){
	            	continue;
	            }
	            
	            for(int j=0;j<labelNum;j++){
	            	if(ins.get(insIndex).stringValue(labelIndices[j]).equals("1")){
	            		c1[j]--;
					}
	            }
	            majBags.get(i).remove(k);
	            deleteInsIndexSet.add(insIndex);
	            deleteInsNum--;
	            
	            if(deleteInsNum<=0){
	            	break;
	            }
	        }
		}
		
		
		Instances ins2=new Instances(ins, 0);
		ins2.delete();
		for(int i=0;i<ins.numInstances();i++){
			if(!deleteInsIndexSet.contains(i)){
				ins2.add(ins.get(i));
			}
		}
		newMlDataset=new MultiLabelInstances(ins2, mlDataset.getLabelsMetaData());
		
		return newMlDataset;
	}
	

}
