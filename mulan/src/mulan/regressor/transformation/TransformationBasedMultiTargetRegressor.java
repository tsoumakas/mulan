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
package mulan.regressor.transformation;

import mulan.classifier.MultiLabelLearnerBase;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Base class for multi-target regressors that use a single-target transformation to handle multi-target data.<br>
 * <br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public abstract class TransformationBasedMultiTargetRegressor extends MultiLabelLearnerBase {

    private static final long serialVersionUID = 1L;
    /**
     * The underlying single-target regressor.
     */
    protected Classifier baseRegressor;

    /**
     * Creates a new instance of {@link TransformationBasedMultiTargetRegressor} with default {@link ZeroR}
     * base regressor.
     */
    public TransformationBasedMultiTargetRegressor() {
        this(new ZeroR());
    }

    /**
     * Creates a new instance.
     * 
     * @param baseRegressor the base regressor which will be used internally to handle the data.
     */
    public TransformationBasedMultiTargetRegressor(Classifier baseRegressor) {
        this.baseRegressor = baseRegressor;
    }

    /**
     * Returns the {@link Classifier} which is used internally by the learner.
     * 
     * @return the internally used regressor
     */
    public Classifier getBaseRegressor() {
        return baseRegressor;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed information about the
     * technical background of this class, e.g., paper reference or book this class is based on.
     * 
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR,
                "Spyromitros-Xioufis, Eleftherios and Tsoumakas, Grigorios and Groves, William and Vlahavas, Ioannis");
        result.setValue(Field.TITLE, "Multi-label Classification Methods for Multi-target Regression");
        result.setValue(Field.JOURNAL, "ArXiv e-prints");
        result.setValue(Field.URL, "http://arxiv.org/abs/1211.6581");
        result.setValue(Field.YEAR, "2014");
        return result;
    }

    /**
     * Returns a string representation of the multi-target regression model by calling
     * {@link #getModelForTarget(int)} for each target. Should always by called after the model has been
     * initialized.
     * 
     * @return a string representation of the multi-target regression model
     */
    public String getModel() {
        if (!isModelInitialized()) {
            return "No model built yet!";
        }
        String modelSummary = "";
        // get the model built for each target
        for (int i = 0; i < numLabels; i++) {
            modelSummary += "\n-- Model for target " + labelNames[i] + ":\n";
            modelSummary += getModelForTarget(i);
        }
        return modelSummary;
    }

    /**
     * Returns a string representation of the single-target regression model build for the target with this
     * targetIndex.
     * 
     * @param targetIndex the target's index
     * @return a string representation of the single-target regression model
     */
    protected abstract String getModelForTarget(int targetIndex);
}
