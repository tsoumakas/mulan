package mulan.sampling;

import java.util.Random;

import mulan.classifier.MultiLabelLearner;

public class ImUtil {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	
	/**
	 * @param min: minimal number in the range
	 * @param max: maximal number in the range
	 * @param rand: a Random object to generate random number 
	 * @return a random int number in the range of [min, max]
	 */
	public static int randomInt(int min, int max,Random rand) {
		if(min>max){
			return -1;
		}
		else if(min==max){
			return min;
		}
		else{
			return Math.abs(rand.nextInt())%(max-min+1)+min;
		}
	}
	
	
	/**
	 * Returns the average of decimals in an array
	 * 
	 * @param A double array 
	 * @return the average of decimals in A
	 */
	public static double getSum(double A[]){
		if(A==null||A.length==0){
			return 0;
		}
		
		double d=0.0;
		for(int i=0;i<A.length;i++){
			d+=A[i];
		}
		return d;
	}
	
	
	/**
	 * Returns the average of decimals in an array
	 * 
	 * @param A double array 
	 * @return the average of decimals in A
	 */
	public static double getAverage(double A[]){
		return getSum(A)/A.length;
	}
	
	/**
	 * Returns the name of a multi-label sampling method
	 * 
	 * @param  sampling a multiLabel sampling method
	 * @return the name of the multiLabel sampling method
	 **/
	public static String getSamplingName(MultiLabelSampling sampling) {
		String s = sampling.getClass().toString(); // s="class mulan.classifier.lazy.MLkNN"
		int Si = s.lastIndexOf(".") + 1;
		int Ei = s.length();
		return s.substring(Si, Ei); // classifierName="MLkNN" 
	}

}
