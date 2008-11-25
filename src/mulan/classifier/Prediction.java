package mulan.classifier;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import weka.core.Utils;

/**
 * Simple container class for multilabel classification result 
 */

public class Prediction {

	protected double[] confidences;
	protected double[] predictedLabels;

	public Prediction(double[] labels, double[] confidences)
	{
		this.confidences = confidences;
		predictedLabels = labels;
	}
        
        
	public double[] getConfidences()
	{
		return confidences;
	}
	
	public double[] getPredictedLabels()
	{
		return predictedLabels;
	}
	
	public boolean getPrediction(int label)
	{
		return Utils.eq(1, predictedLabels[label]);
	}
	
        /**
         * 
         * @param label: the index of a label
         * @param value: whether this label is predicted as true or not
         */
        public void setPrediction(int label, boolean value) 
        {
                if (value)
                    predictedLabels[label] = 1;
                else
                    predictedLabels[label] = 0;
        }   
        
        
	/**
	 * @param label: the index of a label
	 * @return the confidence of the truth (1) of this label
	 */
	public double getConfidence(int label)
	{
		return confidences[label];
	}
                
	
	/**
	 * Number of predicted labels for this instance. 
	 * Calculated only once.
	 */
	protected int numLabels = -1;
	
	/**
	 * Number of predicted labels for this instance.
	 */
	public int numLabels()
	{
		if (numLabels == -1) numLabels =(int) Utils.sum(predictedLabels);
		return numLabels; 
	}

	/**
	 * String representation of the set of labels. Perhaps we
	 * could obtain the actual attribute names from somewhere?
	 */
    @Override
	public String toString()
	{
		StringBuilder str = new StringBuilder().append("{");
		

		str.append("}");
		return str.toString();
	}
	
}
