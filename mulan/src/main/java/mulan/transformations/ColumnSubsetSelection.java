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
package mulan.transformations;

import java.io.Serializable;
import java.util.Random;
import mulan.core.MulanRuntimeException;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>Class that solves the column subset selection problem via random sampling.
 * For more information, see <em>Bi, W., Kwok, J. (2013) Efficient Multi-label
 * Classification with Many Labels, JMLR W&quot;CP 28(3):405-413, 2013</em></p>
 *
 * @author Sotiris L Karavarsamis
 * @author Grigorios Tsoumakas
 * @version 2013.7.16
 */
public class ColumnSubsetSelection implements Serializable {

    private Instances transformed;
    private Remove remove;
    private Matrix Y;
    private Matrix Yc;
    private Matrix ProjectionMatrix;
    private int kappa;
    private int[] selectedIndicesInt;
    private Object[] sampledIndicesObj;
    private static int[] indicesToRemove;
    private java.util.Set sampledIndiceSet;   

    public MultiLabelInstances transform(MultiLabelInstances data, int kappa, long seed) {
        try {

            if (kappa >= data.getNumLabels()) {
                throw new MulanRuntimeException("Dimensionality reduction parameter should not exceed or be equal to the total count of labels!");
            }

            // integer indices of physical label assignments
            int[] labelIndices = data.getLabelIndices();
            int[] indices = new int[labelIndices.length];

            System.arraycopy(labelIndices, 0, indices, 0, labelIndices.length);

            // load label indicator matrix in a Matrix object
            double[][] datmatrix = new double[data.getDataSet().numInstances()][labelIndices.length];
            Matrix mat = new Matrix(datmatrix);

            for (int i = 0; i < data.getDataSet().numInstances(); i++) {
                Instance instance = data.getDataSet().instance(i);
                for (int j = 0; j < labelIndices.length; j++) {
                    mat.set(i, j, Double.parseDouble(instance.toString(labelIndices[j])));
                    //DEBUG: System.out.print("" + Double.parseDouble(instance.toString(labelIndices[j])) + ",");
                }
            }

            // make private copy of the label matrix
            this.Y = mat;

            // compute eigenvalue analysis of label indicator matrix
            SingularValueDecomposition svd = new SingularValueDecomposition(mat);

            //DEBUG: System.out.println("rows = " + svd.getV().getRowDimension() + ", cols = " + svd.getV().getColumnDimension());

            assert (svd.getV().getRowDimension() == svd.getV().getColumnDimension());

            Matrix rVec = svd.getV();
            Matrix Vk = new Matrix(new double[svd.getV().getRowDimension()][kappa]);

            // snippet (2)
            for (int i = 0; i < kappa; i++) {
                for (int j = 0; j < svd.getV().getColumnDimension(); j++) {
                    Vk.set(j, i, rVec.get(i, j));
                }
            }

            // compute column selection probabilitites
            double[] selectionProbabilities = new double[Vk.getRowDimension()];
            double[] selectionProbabilitiesCDF = new double[Vk.getRowDimension()];

            for (int i = 0; i < Vk.getRowDimension(); i++) {
                selectionProbabilities[i] = 0.0;
                for (int j = 0; j < kappa; j++) {
                    selectionProbabilities[i] += Math.pow(Vk.get(i, j), 2);
                }
                selectionProbabilities[i] = Math.sqrt(selectionProbabilities[i]);
            }

            // normalize probabilities
            double psum = 0.0;
            for (int i = 0; i < Vk.getRowDimension(); i++) {
                psum += selectionProbabilities[i];
                //System.out.println("psum = " + psum);
            }
            //System.out.println("psum = " + psum);
            //assert (psum != 0 && psum == 1.0); // must be non-zero and unitary

            for (int i = 0; i < Vk.getRowDimension(); i++) {
                selectionProbabilities[i] /= psum;
            }

            psum = 0.0;
            for (int i = 0; i < Vk.getRowDimension(); i++) {
                psum += selectionProbabilities[i];
                selectionProbabilitiesCDF[i] = psum;
            }

            // add selected columns on a linked list
            sampledIndiceSet = new java.util.HashSet();

            // run column-sampling loop
            int sampling_count = 0;
            
            Random generator = new Random(seed);
            while (sampledIndiceSet.size() < kappa) // ...loop until knapsack gets filled...
            {
                // pick a random number

                //DEBUG:
                //double roulette = generator.nextDouble() * 0.5;
                double roulette = generator.nextDouble();

                // seek closest match according to sampling probabilities
                int closest_match = -1;

                // iterate label cols
                for (int i = 0; i < Vk.getRowDimension(); i++) {
                    if (roulette < selectionProbabilitiesCDF[i]) // ...spot a possible match...
                    {
                        // ...if so, select and quit scope...
                        closest_match = i; // BEWARE! "i" is an index over the label enumeration, not an ordering index!
                        break;
                    }
                }

                // if we stepped on the flag, something serious is going on!
                assert (closest_match != -1);

                // see if column was selected; if not, add it
                if (!sampledIndiceSet.contains((Object) closest_match)) {
                    sampledIndiceSet.add((Object) closest_match);
                    //System.out.println("DEBUG(CSSP): Added column " + closest_match + " to the sampled column set!");
                }

                sampling_count += 1;
            }

            System.out.println("Sampling loop completed in " + sampling_count + " runs.");

            // compute indices-to-remove array
            indicesToRemove = new int[labelIndices.length - sampledIndiceSet.size()];

            // compute all **PHYSICAL** (not VIRTUAL) indices of label columns for CSSP to remove
            int idx = 0;
            for (int i = 0; i < labelIndices.length; i++) {
                if (!sampledIndiceSet.contains((Object) i)) {
                    indicesToRemove[idx] = indices[i];
                    idx += 1;
                }
            }

            // apply CSSP: select columns to remove
            int[] selectedIndicesObj = indicesToRemove.clone();
            selectedIndicesInt = new int[selectedIndicesObj.length];
            for (int i = 0; i < selectedIndicesObj.length; i++) {
                selectedIndicesInt[i] = (int) selectedIndicesObj[i];
            }

            // compute Moore-Penrose pseudo-inverse matrix of the column-reduced label indicator matrix
            double[][] datmatrix2 = new double[data.getDataSet().numInstances()][labelIndices.length - selectedIndicesInt.length];
            Matrix matC = new Matrix(datmatrix2);

            //DEBUG:
            //System.out.println("Selecting only " + matC.getColumnDimension() + " columns; removing " + selectedIndicesInt.length + " columns out of an original total of " + data.getLabelIndices().length + " labels!");

            // compute indices to keep
            java.util.LinkedList<Integer> indicesToKeep = new java.util.LinkedList();
            for (int i = 0; i < labelIndices.length; i++) {
                boolean keep = true;

                // see if this col has to be removed
                for (int k = 0; k < selectedIndicesInt.length; k++) {
                    if (selectedIndicesInt[k] == labelIndices[i]) {
                        keep = false;
                        break;
                    }
                }

                // add if we actually should keep this...
                if (keep) {
                    indicesToKeep.add(labelIndices[i]);
                }
            }

            assert (indicesToKeep.size() == matC.getColumnDimension());

            for (int i = 0; i < matC.getRowDimension(); i++) {
                // get data instance
                Instance instance = data.getDataSet().instance(i);

                // replicate data from ALL columns that WOULD not be removed by CSSP        	
                for (int j = 0; j < matC.getColumnDimension(); j++) {
                    // get label indice
                    int corrIdx = (int) indicesToKeep.get(j);

                    // update matC
                    matC.set(i, j, Double.parseDouble(instance.toString(corrIdx)));
                }
            }

            //DEBUG: System.out.println("matC rows = " + matC.getRowDimension() + ", cols = " + matC.getColumnDimension() + "\n data original label cols # = " + data.getLabelIndices().length);

            // make private copy of projection matrices

            // Moore-Penrose pseudo-inverse of the label matrix matC
            // see http://robotics.caltech.edu/~jwb/courses/ME115/handouts/pseudo.pdf for an SVD-based workaround for MP-inverse

            // Moore-Penrose pseudoinverse computation based on Singular Value Decomposition (SVD)
            /*
             SingularValueDecomposition decomp = Vk.svd();
            
             Matrix S = decomp.getS();
             Matrix Scross = new Matrix(selectedIndicesInt.length,selectedIndicesInt.length);
             for(int i = 0; i < selectedIndicesInt.length; i++) {
             for(int j = 0; j < selectedIndicesInt.length; j++) {
             if(i == j) {
             if(S.get(i, j) == 0) {
             Scross.set(i, j, 0.0);
             } else {
             Scross.set(i, j, 1 / S.get(i, j));
             }
             } else {
             Scross.set(i, j, 0.0);
             }
             }
             }
            
             this.Yc = decomp.getV().times(Scross).times(decomp.getU().transpose());
             */

            // DEBUG: traditional way of computing the Moore-Penrose pseudoinverse
            if (matC.getRowDimension() >= matC.getColumnDimension()) {
                this.Yc = ((matC.transpose().times(matC)).inverse()).times(matC.transpose());
            } else {
                this.Yc = matC.transpose().times((matC.times(matC.transpose()).inverse()));
            }

            //System.out.println("Yc rows: " + Yc.getRowDimension() + "\nYc cols: " + Yc.getColumnDimension() + "\n Y rows: " + Y.getRowDimension() + "\nY cols: " + Y.getColumnDimension());

            this.ProjectionMatrix = Yc.times(Y); // compute projection matrix

            // add sampled indices to Remove object
            remove = new Remove();
            remove.setAttributeIndicesArray(selectedIndicesInt);
            remove.setInvertSelection(false);
            remove.setInputFormat(data.getDataSet());

            // apply remove filter on the labels
            transformed = Filter.useFilter(data.getDataSet(), remove);

            this.sampledIndicesObj = indicesToKeep.toArray();
            
            return data.reintegrateModifiedDataSet(transformed);

        } catch (Exception ex) {
            // do nothing
            //Logger.getLogger(BinaryRelevanceTransformation.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }

    /**
     * getMaxSampledLabels
     *
     * @return maximal number of labels to sample out of a pool of original
     * labels
     */
    public int getMaxSampledLabels() {
        return kappa;
    }

    /**
     * getProjectionMatrix
     *
     * @return the sub-space projection matrix of labels
     */
    public Matrix getProjectionMatrix() {
        return ProjectionMatrix;
    }

    /**
     * Get sampled label indices
     *
     * @return sampled label integer indices
     */
    public Object[] getSampledIndices() {
        return this.sampledIndicesObj;
    }

    /**
     * Remove labels not being contained in the sampled label set given a data
     * instance
     *
     * @param instance the instance from which the labels are to be removed
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance) {
        Instance transformedInstance;

        remove.input(instance);
        transformedInstance = remove.output();

        return transformedInstance;
    }
}