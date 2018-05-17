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
package mulan.evaluation.measure;

/**
 * Class for computing various information retrieval measures.
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.29
 */
public class InformationRetrievalMeasures {

    /**
     * Computation of F-measure based on tp, fp, fn and beta. We treat special
     * cases with empty set predictions or/and ground truth as follows: (i) if
     * the algorithm outputs the empty set and the ground truth is the empty
     * set, then we consider F equal to 1 (ii) if the algorithm outputs the
     * empty set and the ground truth is not empty, then we consider F equal to
     * 0 (iii) if the ground truth is empty and the algorithm does not output
     * the empty set, then we consider F equal to 0 (iv) if neither the ground
     * truth nor the algorithm's prediction is the empty set and their
     * intersection is empty, then we consider F equal to 0.
     *
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @param beta controls the relative importance of recall versus precision
     * @return the value of the f-measure
     */
    public static double fMeasure(double tp, double fp, double fn, double beta) {
        if (tp + fp + fn == 0) {
            return 1;
        }
        double beta2 = beta * beta;
        return ((beta2 + 1) * tp) / ((beta2 + 1) * tp + beta2 * fn + fp);
    }

    /**
     * Computation of precision based on tp, fp and fn. We treat special cases
     * with empty set predictions or/and ground truth as follows: (i) if the
     * algorithm outputs the empty set and the ground truth is the empty set,
     * then we consider precision equal to 1 (ii) if the algorithm outputs the
     * empty set and the ground truth is not empty, then we consider precision
     * equal to 0.
     *
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @return the value of precision
     */
    public static double precision(double tp, double fp, double fn) {
        if (tp + fp + fn == 0) {
            return 1;
        }
        if (tp + fp == 0) {
            return 0;
        }
        return tp / (tp + fp);
    }

    /**
     * Computation of recall based on tp, fp and fn. We treat special cases with
     * empty set predictions or/and ground truth as follows: (i) if the
     * algorithm outputs the empty set and the ground truth is the empty set,
     * then we consider recall equal to 1 (ii) if the algorithm does not output
     * the empty set and the ground truth is empty, then we consider recall
     * equal to 0.
     *
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @return the value of recall
     */
    public static double recall(double tp, double fp, double fn) {
        if (tp + fp + fn == 0) {
            return 1;
        }
        if (tp + fn == 0) {
            return 0;
        }
        return tp / (tp + fn);
    }

    /**
     * Computation of specificity based on tn, fp and fn. We treat special cases
     * with empty set predictions or/and ground truth as follows: (i) if the
     * algorithm outputs the set of all labels and the ground truth is the set
     * of all labels, then we consider specificity equal to 1 (ii) if the ground
     * truth is the set of all labels and the algorithm does not output the set
     * of all labels, then we consider specificity equal to 0.
     *
     * @param tn true negatives
     * @param fp false positives
     * @param fn false negatives
     * @return the value of specificity
     */
    public static double specificity(double tn, double fp, double fn) {
        if (tn + fp + fn == 0) {
            return 1;
        }
        if (tn + fp == 0) {
            return 0;
        }
        return tn / (tn + fp);
    }
}