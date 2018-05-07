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
package mulan.classifier.meta;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.core.MulanRuntimeException;
import mulan.data.MultiLabelInstances;
import mulan.transformations.ColumnSubsetSelection;
import weka.core.matrix.Matrix;
import weka.core.Instance;


/**
 * <p>Algorithm for selecting a small subset of labels that can approximately 
 * span the original label space. For more information see: <em>Bi, W., Kwok, J. 
 * (2013) Efficient Multi-label Classification with Many Labels, JMLR W&quot;CP
 * 28(3):405-413, 2013</em></p>
 *
 * @author Sotiris L Karavarsamis
 * @author Grigorios Tsoumakas
 * @version 2013.07.15
 */
public class MLCSSP extends MultiLabelMetaLearner {

    /**
     * The correspondence between ensemble models and labels
     */
    private int kappa;
    private ColumnSubsetSelection css;

    /**
     * Constructs a learner object
     * 
     * @param learner the underlying multi-label learner
     * @param aKappa the number of labels to keep
     */
    public MLCSSP(MultiLabelLearner learner, int aKappa) {
        super(learner);
        kappa = aKappa;
    }

    /**
     * Constructs a learner object. The number of labels to keep will be set to 
     * 0.1 of the total number of labels
     * 
     * @param learner the underlying multi-label learner.
     */
    public MLCSSP(MultiLabelLearner learner) {
        super(learner);
        kappa = 0;
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        // check whether labels have missing values
        for (int i=0; i< numLabels; i++) {
            if (train.getDataSet().attributeStats(labelIndices[i]).missingCount > 0) {
                throw new MulanRuntimeException("Algorithm does not work when labels have missing values");
            }
        }
        
        // autoselect 10% of labels of label count is not defined
        if (kappa == 0) {
            this.kappa = (int) Math.round(0.1 * (double) train.getNumLabels());
        }

        // we need at least two label attributes
        if (kappa <= 1) {
            kappa = 2;
        }
        debug("kappa = " + kappa);

        css = new ColumnSubsetSelection();
        MultiLabelInstances transformed = css.transform(train, kappa, 1);
        baseLearner.build(transformed);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) {
        try {

            Instance transformed = css.transformInstance(instance);
            MultiLabelOutput out = baseLearner.makePrediction(transformed);

            double[] confidences = out.getConfidences();

            // make response matrix
            Matrix conf = new Matrix(kappa, 1);
            for (int i = 0; i < kappa; i++) {
                conf.set(i, 0, confidences[i]);
            }

            // compute projected classifier response
            Matrix projectedResponse = conf.transpose().times(css.getProjectionMatrix());

            boolean[] projected_bipartition = new boolean[projectedResponse.getColumnDimension()];
            double[] projected_confidences = new double[projectedResponse.getColumnDimension()];

            for (int i = 0; i < projectedResponse.getColumnDimension(); i++) {
                projected_confidences[i] = projectedResponse.get(0, i);
                projected_bipartition[i] = (Math.ceil(projected_confidences[i]) == 1) ? true : false;
            }

            // return mlo
            MultiLabelOutput mlo = new MultiLabelOutput(projected_bipartition, projected_confidences);
            return mlo;

        } catch (InvalidDataException ex) {
            Logger.getLogger(MLCSSP.class.getName()).log(Level.SEVERE, null, ex);

        } catch (ModelInitializationException ex) {
            Logger.getLogger(MLCSSP.class.getName()).log(Level.SEVERE, null, ex);

        } catch (Exception ex) {
            Logger.getLogger(MLCSSP.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }

}