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

/*
 *    UnconditionalChiSquareIdentifier.java
 */
package mulan.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class for identification of unconditional dependence between each pair of labels using Chi Square Test For Independence.
 * The chi-square test for independence is applied to the number of instances for each possible combination of two categories. 
 *
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public class UnconditionalChiSquareIdentifier implements LabelPairsDependenceIdentifier,  Serializable {

    /** A default chi square critical value, corresponds to significance level 0.01. Label pairs with dependence value below the critical are considered as independent.*/
    private double criticalValue = 6.635;

    /**
     * Calculates Chi Square values for each pair of labels.  It uses Phi correlation value calculated in {@link mulan.data.Statistics} as follows: ChiSquareValue = PhiValue^2 * NumberOfInstances in the data set. 

     * @param mlInstances multilabel data set
     * @return an array of label pairs sorted in descending order of the ChiSquare value
     */
    public LabelsPair[] calculateDependence(MultiLabelInstances mlInstances){
        LabelsPair[] pairs;
        List<LabelsPair> chiPairsList = new ArrayList<LabelsPair>();
        double v;
        Statistics st = new Statistics();
        int N= mlInstances.getDataSet().numInstances();
        try {
            double[][] matrix = st.calculatePhi(mlInstances);
            for(int i=0; i<matrix.length-1; i++){
                for(int j=i+1; j<matrix[i].length; j++){
                    int[] pair = new int[2];
                    pair[0] = i;
                    pair[1] = j;
                    double val = matrix[i][j];
                    if(Double.isNaN(val)){
                        v=0.0001;
                    }
                    else{
                        v= Math.pow(val,2)*N;
                    }
                    chiPairsList.add(new LabelsPair(pair, v));
                }
            }
        } catch (Exception e) {
            Logger.getLogger(UnconditionalChiSquareIdentifier.class.getSimpleName()).log(Level.SEVERE, null, e);
        }
        finally{
            pairs = new LabelsPair[chiPairsList.size()];
            chiPairsList.toArray(pairs);
            Arrays.sort(pairs, Collections.reverseOrder());
        }
        return pairs;
    }

    /**
     * 
     * @param criticalValue the critical value to set
     */
    public void setCriticalValue(double criticalValue) {
        this.criticalValue = criticalValue;
    }

    public double getCriticalValue() {
        return criticalValue;
    }

}