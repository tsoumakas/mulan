package mulan.dimensionalityReduction;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class SimpleMostFrenquencyReduction {
	protected double percentRetainFeatures;
	protected int[] retainFeatureIndices=null;
	
	public SimpleMostFrenquencyReduction() {
		this.percentRetainFeatures=0.1;
	}
	
	public SimpleMostFrenquencyReduction(double percentRetainFeatures){
		if(percentRetainFeatures>0&&percentRetainFeatures<=1){
			this.percentRetainFeatures=percentRetainFeatures;
		}
		else{
			this.percentRetainFeatures=0.1;
		}
	}
	
	public double getPercentRetainFeatures() {
		return percentRetainFeatures;
	}
	
	public int[] getRetainFeatureIndices() {
		return retainFeatureIndices;
	}
	
	
	protected void obtainFetainedfeatureIndices(MultiLabelInstances mlData){
		int featureIndices[]=mlData.getFeatureIndices();
		int numFeatures=featureIndices.length;
		int numRetainFeatures=(int)(numFeatures*percentRetainFeatures);
		retainFeatureIndices=new int[numRetainFeatures];
		
		Instances ins=mlData.getDataSet();
		
		int cs[]=new int[numFeatures];
		Arrays.fill(cs, 0);
		for(Instance data:ins){
			for(int i=0;i<numFeatures;i++){
				int index=featureIndices[i];
				if(data.attribute(index).isNominal()){
					if(!data.stringValue(index).equals("0")){
						cs[i]++;
					}
				}
				else if(data.attribute(index).isNumeric()){
					if(data.value(index)!=0.0D){
						cs[i]++;
					}
				}
				else{
					System.out.println("Can not deal with the not nominal or numeric tyep attribute (attribute index: "+index+")");
				}
			}
		}
		
		HashMap<Integer,ArrayList<Integer>> map=new HashMap<>(); //<number of none zero values, list of feature index>
		for(int i=0;i<numFeatures;i++){	
			if(!map.containsKey(cs[i])){
				ArrayList<Integer> list=new ArrayList<>();
				list.add(featureIndices[i]);
				map.put(cs[i],list);
			}
			else{
				ArrayList<Integer> list=map.get(cs[i]);
				list.add(featureIndices[i]);
				map.put(cs[i],list);
			}
		}
		
		//Sort the map according the number of none zero values
		List<Map.Entry<Integer,ArrayList<Integer>>> list = new ArrayList<Map.Entry<Integer,ArrayList<Integer>>>(map.entrySet());
        Collections.sort(list,new Comparator<Map.Entry<Integer,ArrayList<Integer>>>() {
            //Descending Order
            public int compare(Entry<Integer,ArrayList<Integer>> o1,
                    Entry<Integer,ArrayList<Integer>> o2) {
                return -1*o1.getKey().compareTo(o2.getKey());
            }   
        });
        int c=0;
        boolean isBreak=false;
        for (Iterator iter = list.iterator(); iter.hasNext();){  
        	Map.Entry entry = (Map.Entry)iter.next();  
        	//System.out.println(entry.getKey()+"\t"+Arrays.asList(((ArrayList<Integer>)entry.getValue())).toString());
        	
        	for(int index:((ArrayList<Integer>)entry.getValue())){
        		retainFeatureIndices[c++]=index;
        		if(c>=numRetainFeatures){
        			isBreak=true;
        			break;
        		}
        	}
        	if(isBreak){
        		break;
        	}
        	
        }  
	}

	public MultiLabelInstances build(MultiLabelInstances mlData) throws Exception{
		obtainFetainedfeatureIndices(mlData);
		
		int numFeatures=mlData.getFeatureIndices().length;
		int numRetainFeatures=(int)(numFeatures*percentRetainFeatures);
		
		int[] allRetainAttributes = new int[numRetainFeatures + mlData.getNumLabels()];
        System.arraycopy(retainFeatureIndices, 0, allRetainAttributes, 0, numRetainFeatures);
        int[] labelIndices = mlData.getLabelIndices();
        System.arraycopy(labelIndices, 0, allRetainAttributes, numRetainFeatures, mlData.getNumLabels());

        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(allRetainAttributes);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(mlData.getDataSet());
        Instances filtered = Filter.useFilter(mlData.getDataSet(), filterRemove);
        MultiLabelInstances mlFiltered = new MultiLabelInstances(filtered, mlData.getLabelsMetaData());
        
        return mlFiltered;
	}

}
