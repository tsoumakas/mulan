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
 *    LabelPowersetAttributeEvaluator.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.attributeSelection;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.core.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import mulan.transformations.multiclass.MultiClassTransformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 * Performs attribute evaluation using single-label transformations. For
 * more information, see <br/>
 * <br/>
 * Chen, W., Yan, J., Zhang, B., Chen, Z., and Yang, Q. (2007).
 * Document transformation for multi-label feature selection in text categorization.
 * In 7th IEEE International Conference on Data Mining (ICDM'07), pages 451-456.
 * </p>
 *
 * BibTeX:
 *
 * <pre>
 * &#064;inproceedings{chen+etal:2007,
 * 	author = {Chen, Weizhu and Yan, Jun and Zhang, Benyu and Chen, Zheng and Yang, Qiang},
 *  booktitle = {Proc. 7th IEEE International Conference on Data Mining (ICDM'07)},
 *  pages = {451--456},
 *  title = {Document Transformation for Multi-label Feature Selection in Text Categorization},
 *  year = {2007}
 * }
 * </pre>
 *
 * @author Grigorios Tsoumakas
 */
public class MultiClassAttributeEvaluator extends ASEvaluation implements AttributeEvaluator
{
   /** The single-label attribute evaluator to use underneath */
    private ASEvaluation baseAttributeEvaluator;

    /** Constructor that uses an evaluator on a multi-label dataset using a transformation */
    public MultiClassAttributeEvaluator(ASEvaluation x, MultiClassTransformation dt, MultiLabelInstances mlData)
    {
        baseAttributeEvaluator = x;
        Instances data;
        try {
            data = dt.transformInstances(mlData);
            System.out.println(data);
            ((ASEvaluation) baseAttributeEvaluator).buildEvaluator(data);
        } catch (Exception ex) {
            Logger.getLogger(MultiClassAttributeEvaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return ((AttributeEvaluator) baseAttributeEvaluator).evaluateAttribute(attribute);
    }

    @Override
    public void buildEvaluator(Instances arg0) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }


}
