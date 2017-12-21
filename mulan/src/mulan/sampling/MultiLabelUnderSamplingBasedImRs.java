package mulan.sampling;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.hypernet.BaseFunction;
import mulan.core.Util;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

public class MultiLabelUnderSamplingBasedImRs extends MultiLabelSampling{
	
	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception{
		rnd=new Random(seed);
		
		MultiLabelInstances newMlDataset=null;
		Instances ins=mlDataset.getDataSet();
		
		int oriInsNum=mlDataset.getNumInstances();
		int deleteInsNum=(int)(P*oriInsNum);
		int labelNum=mlDataset.getNumLabels();
		int labelIndices[]=mlDataset.getLabelIndices();
		
		
		
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateImSta(mlDataset);
		double ImRs[]=is.getIRLbls();
		int c0[]=is.getC0();
		int c1[]=is.getC1();
		
		
		HashSet<Integer> retainInsIndexSet=new HashSet<Integer>();
		for(int i=0;i<oriInsNum;i++){
			retainInsIndexSet.add(i);
		}
		
		
		ArrayList<Integer> canDeleteList=new ArrayList<Integer>();

		
		while(deleteInsNum>0){
			int maxIndex=Util.RandomIndexOfMax(ImRs, rnd);
			double minsImR=Double.MAX_VALUE;
			canDeleteList.clear();
			
			for(int i:retainInsIndexSet){
				Instance data=ins.get(i);
				if(data.stringValue(labelIndices[maxIndex]).equals("0")){
					for(int j=0;j<labelIndices.length;j++){
						if(data.stringValue(labelIndices[j]).equals("1")){
							c1[j]--;
						}
						else{
							c0[j]--;
						}
					}
					
					double d=getsumImR(c0, c1);
					if(Math.abs(d-minsImR)<1e-10){
						canDeleteList.add(i);
					}
					else if(d<minsImR){
						minsImR=d;
						canDeleteList.clear();
						canDeleteList.add(i);
					}
				}
				for(int j=0;j<labelIndices.length;j++){
					if(data.stringValue(labelIndices[j]).equals("1")){
						c1[j]++;
					}
					else{
						c0[j]++;
					}
				}
			}
			
			
			if(canDeleteList.size()==0){
				System.out.println("!!!!");
			}
			int deleteIndex;
			if(canDeleteList.size()==1){
				deleteIndex=canDeleteList.get(0);
			}
			else{
				int k=ImUtil.randomInt(0, canDeleteList.size()-1, rnd);
				deleteIndex=canDeleteList.get(k);
			}
			
			retainInsIndexSet.remove(deleteIndex);
			Instance data=ins.get(deleteIndex);
			for(int j=0;j<labelIndices.length;j++){
				if(data.stringValue(labelIndices[j]).equals("1")){
					c1[j]--;
				}
				else{
					c0[j]--;
				}
				
				if(c1[j]==0){	
	        		ImRs[j]=Double.MAX_VALUE;
	        	}
	        	else{
	        		ImRs[j]=c0[j]*1.0/c1[j];
	        	}
			}
			deleteInsNum--;
		}
		
		
		Instances ins2=new Instances(ins, 0);
		ins2.delete();
		for(int i:retainInsIndexSet){
			ins2.add(ins.get(i));
		}
		newMlDataset=new MultiLabelInstances(ins2, mlDataset.getLabelsMetaData());
		
		return newMlDataset;
	}
	
	
	private double getsumImR(int c0[],int c1[]){
		if(c0==null || c1==null || c0.length!=c1.length)
			return Double.MAX_VALUE;
		double s=0;
		for(int i=0;i<c0.length;i++){
			if(c1[i]==0){
				s+=(c0[i]*1.0/0.1);  ////!!!!!!
			}else{
				s+=(c0[i]*1.0/c1[i]);  
			}
			
		}
		return s;
	}

}
