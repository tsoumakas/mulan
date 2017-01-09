package MultiLabelHyperNetWork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

import MultilabelCompetition.BaseFuction;
import weka.core.Instance;

public class HyperEdge {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	private ArrayList<Boolean> valueTypeList=null;  //true denotes Integer ; false denotes Double
	private ArrayList<Double> valueList=null;
	private ArrayList<Integer> vertexList=null;
	private ArrayList<Integer> labelList=null;
	private ArrayList<Double> weightList=null;
	int order=-1;
	int classIndex=-1;
	Double fitness=Double.MIN_VALUE;
	

	
	public int getLabel(int labelIndex){
		return labelList.get(labelIndex);
	}
	
	public double getWeight(int labelIndex){
		return weightList.get(labelIndex);
	}
	
	public Double getFitness(){
		return this.fitness;
	}
	
	public void setWeight(double weight,int index){
		this.weightList.set(index, weight);
	}
	
	public void setFitness(double fitness){
		this.fitness=fitness;
	}
	
	
	
	
	
	HyperEdge(){
		
	}
	
	HyperEdge(Double valueArray[] ,Integer vertexArray[],Integer[] labelArray,int classIndex) throws Exception{
		if(valueArray.length!=vertexArray.length)
			return;
		
		order=vertexArray.length;
		this.classIndex=classIndex;
		fitness=Double.MIN_VALUE;
		
		valueTypeList=new ArrayList<Boolean>();	
		for(int i=0;i<order;i++){
			valueTypeList.add(true);
		}
		
		valueList=BaseFuction.getArrayList(valueArray);
		vertexList=BaseFuction.getArrayList(vertexArray);
		labelList=BaseFuction.getArrayList(labelArray);		
		weightList=new ArrayList<Double>();
		for(int i=0;i<labelArray.length;i++){
			weightList.add(1.0);
		}		
	}
	
	HyperEdge(Instance data ,Integer vertexArray[],int labelNum,int classIndex) throws Exception{
		if(vertexArray.length<1)
			return;
		
		order=vertexArray.length;
		this.classIndex=classIndex;
		fitness=Double.MIN_VALUE;
		
		valueTypeList=new ArrayList<Boolean>();
		for(int i=0;i<order;i++){
			valueTypeList.add(true);
		}
		
		vertexList=BaseFuction.getArrayList(vertexArray);
		
		valueList=new ArrayList<Double>();
		for(int i=0;i<order;i++){
			valueList.add(data.value(vertexList.get(i)));
		}
		
		weightList=new ArrayList<Double>();
		for(int i=0;i<labelNum;i++){
			weightList.add(1.0);
		}
		
		labelList=new ArrayList<Integer>();		
		for(int i=2*labelNum;i<3*labelNum;i++){
			labelList.add((int) data.value(i));
		}
	}

	HyperEdge(Instance data ,Integer vertexArray[],Boolean [] valueTypeArray,int labelNum,int classIndex) throws Exception{
		if(vertexArray.length<1)
			return;
		
		order=vertexArray.length;
		this.classIndex=classIndex;
		fitness=Double.MIN_VALUE;
		
		valueTypeList=new ArrayList<Boolean>();
		for(int i=0;i<order;i++){
			valueTypeList.add(valueTypeArray[i]);
		}
		
		vertexList=BaseFuction.getArrayList(vertexArray);
		
		valueList=new ArrayList<Double>();
		for(int i=0;i<order;i++){
			valueList.add(data.value(vertexList.get(i)));	
		}
			
		weightList=new ArrayList<Double>();
		for(int i=0;i<labelNum;i++){
			weightList.add(1.0);
		}
		
		labelList=new ArrayList<Integer>();		
		for(int i=2*labelNum;i<3*labelNum;i++){
			labelList.add((int) data.value(i));
		}
	}

	
	public int[] classifyInstance(Instance data,int LabelNum){
		int results[]=new int [LabelNum];
		
		for(int j=LabelNum;j>0;j--){
			if(Math.abs((data.value(data.numAttributes()-j)-labelList.get(LabelNum-j)))<1e-8){
				results[LabelNum-j]=1;
			}
		}
		
		return results;
	}


	public boolean isMatch(Instance data,double matchThreshold){
		if(order<=0||valueList==null)
			return false;
		boolean flag=true;
		for(int i=0;i<order&&flag;i++){
			if(valueTypeList.get(i)) //Integer
			{
				if(valueList.get(i)!=(data.value(vertexList.get(i)))){
					flag=false;
				}
			}
			else  //Double
			{
				if(Math.abs((valueList.get(i)-(data.value(vertexList.get(i)))))>matchThreshold){
					flag=false;
				}
			}
		}
		return flag;
	}
	
	public void updateWeigth(double weigthChangeArray[]){
/*		if(weightList.size()!=weigthChangeArray.length)
			return;*/
		for(int i=0;i<weightList.size();i++){
			weightList.set(i,weightList.get(i)+weigthChangeArray[i]);
		}
	}
	
	public void updateWeigth(double weigthChange,int index){
		weightList.set(index,weightList.get(index)+weigthChange);
	}	
}
