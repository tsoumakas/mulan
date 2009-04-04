package mulan.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class PhiCoefficient {

	int numOfLabels;
	double[][] phi;

	public double[][] calculatePhi(Instances dataSet, int numOfLabels)
			throws Exception {
		this.numOfLabels = numOfLabels;
		int predictors = dataSet.numAttributes() - numOfLabels;
		phi = new double[numOfLabels][numOfLabels];
		// Indices of label attributes
		int indices[] = new int[numOfLabels];
		int k = 0;
		for (int j = 0; j < numOfLabels; j++) {
			indices[k] = predictors + j;
			k++;
		}

		Remove remove = new Remove();
		remove.setInvertSelection(true);
		remove.setAttributeIndicesArray(indices);
		remove.setInputFormat(dataSet);
		Instances result = Filter.useFilter(dataSet, remove);
		result.setClassIndex(result.numAttributes() - 1);

		for (int i = 0; i < numOfLabels; i++) {
			int a[] = new int[numOfLabels];
			int b[] = new int[numOfLabels];
			int c[] = new int[numOfLabels];
			int d[] = new int[numOfLabels];
			double e[] = new double[numOfLabels];
			double f[] = new double[numOfLabels];
			double g[] = new double[numOfLabels];
			double h[] = new double[numOfLabels];
			for (int j = 0; j < result.numInstances(); j++) {
				for (int l = 0; l < numOfLabels; l++) {
					// if (l == i) {
					// continue;
					// }
					if (result.instance(j).value(i) == 0.0) {
						if (result.instance(j).value(l) == 0.0) {
							a[l]++;
						} else {
							c[l]++;
						}
					} else {
						if (result.instance(j).value(l) == 0.0) {
							b[l]++;
						} else {
							d[l]++;
						}
					}
				}
			}
			for (int l = 0; l < numOfLabels; l++) {
				e[l] = a[l] + b[l];
				f[l] = c[l] + d[l];
				g[l] = a[l] + c[l];
				h[l] = b[l] + d[l];

				double mult = e[l] * f[l] * g[l] * h[l];
				double denominator = Math.sqrt(mult);
				double nominator = a[l] * d[l] - b[l] * c[l];
				phi[i][l] = nominator / denominator;

			}
		}
		return phi;
	}

	public void printCorrelations() {
		String pattern = "0.00";
		DecimalFormat myFormatter = new DecimalFormat(pattern);

		for (int i = 0; i < numOfLabels; i++) {
			for (int j = 0; j < numOfLabels; j++) {
				System.out.print(myFormatter.format(phi[i][j]) + " ");
			}
			System.out.println("");
		}
	}
	
	/**
	 * This method prints data, useful for the visualization of Phi per dataset.
	 * It prints int(1/step) + 1 pairs of values.
	 * The first value of each pair is the phi value and the second is the average
	 * number of labels that correlate to the rest of the labels with correlation
	 * higher than the specified phi value;
	 * 
	 * @param step the phi value increment step
	 */
	public void printDiagram(double step){
		String pattern = "0.00";
		DecimalFormat myFormatter = new DecimalFormat(pattern);
		
		System.out.println("Phi      AvgCorrelated");
		double phi = 0;
		while(phi<=1.001){
			double avgCorrelated=0;
			for(int i=0;i<numOfLabels;i++){
				int [] temp = uncorrelatedIndices(i,phi);
				avgCorrelated += (numOfLabels - temp.length);
			}
			avgCorrelated/=numOfLabels;
			System.out.println(myFormatter.format(phi) + "     " + avgCorrelated);
			phi+= step;
		}
	}
	
	/**
	 * returns the indices of the labels whose phi coefficient
	 * values lie between -bound <= phi <= bound 
	 * 
	 * @param labelIndex
	 * @param bound
	 * @return
	 */
	public int [] uncorrelatedIndices(int labelIndex, double bound){
		ArrayList<Integer>  indiceslist = new ArrayList<Integer>();
		for(int i=0;i<numOfLabels;i++){
			if(Math.abs(phi[labelIndex][i])<=bound){
				indiceslist.add(i);
			}
		}
		int [] indices = new int[indiceslist.size()];
		for(int i =0;i<indiceslist.size();i++){
			indices[i] = indiceslist.get(i);
		}
		//System.out.println(Arrays.toString(indices));
		return indices;
	}
}
