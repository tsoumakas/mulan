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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.console.builder;

import java.util.Arrays;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.neural.MMPLearner;
import mulan.classifier.neural.MMPUpdateRuleType;
import mulan.console.CommonOptions;
import mulan.core.ArgumentNullException;
import mulan.evaluation.loss.RankingLossFunction;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * Command line interface builder for {@link MMPLearner}.
 * 
 * @see LearnerBuilder
 * @author Jozef Vilcek
 */
public class MMPLearnerBuilder implements LearnerBuilder {

	private static final String RANKING_LOSS_OPT = "rl";
	private static final String UPDATE_RULE_OPT = "ur";
	private static final String NOMINAL_TO_BINARY_OPT = "n2b";
	
	@SuppressWarnings("static-access")
	@Override
	public Options getOptions() {
		
		StringBuilder sb = new StringBuilder();
		sb.append("Possible update rule options are: ");
		sb.append("a) ").append(MMPUpdateRuleType.MaxUpdate).append(", ");
		sb.append("b) ").append(MMPUpdateRuleType.RandomizedUpdate).append(", ");
		sb.append("c) ").append(MMPUpdateRuleType.UniformUpdate);
		
		Options opt = new Options();
		opt.addOption(OptionBuilder.withLongOpt("ranking-loss").hasArg()
				.withDescription("Sets loss measure to be used when judging	ranking performance in learning process. The option is required and value must be full name of a class type impelementing the measure.")
				.create(RANKING_LOSS_OPT));
		opt.addOption(OptionBuilder.withLongOpt("update-rule").hasArg()
				.withDescription("Sets the update rule to be used when updating the learner model during the learning phase. " + sb.toString())
				.create(UPDATE_RULE_OPT));
		opt.addOption(OptionBuilder.withLongOpt("nominal-to-binary")
				.withDescription("If defined, all nominal attributes from train data set will be converted to binary prior to learning (and respectively making a prediction).")
				.create(NOMINAL_TO_BINARY_OPT));
		opt.addOption(CommonOptions.getSeedOption());
		
		return opt;
	}

	@Override
	public Class<? extends MultiLabelLearner> getSupportedType(){
		return MMPLearner.class;
	}

	@Override
	public MultiLabelLearner build(CommandLine cmdLine) throws ParseException {
		if(cmdLine == null){
			throw new ArgumentNullException("cmdLine");
		}
		
		RankingLossFunction loss = null;
		if(!cmdLine.hasOption(RANKING_LOSS_OPT)){
			throw new ParseException(String.format("The option '%s' is required.", RANKING_LOSS_OPT));
		}
		String lossName = cmdLine.getOptionValue(RANKING_LOSS_OPT);
		try{
			loss = Class.forName(lossName).asSubclass(RankingLossFunction.class).newInstance();
		}catch(Exception ex){
			throw new ParseException(String.format("The loss measure with class name '%s' can not be created.", lossName));
		}
		
		MMPUpdateRuleType updateRule = null;
		if(!cmdLine.hasOption(UPDATE_RULE_OPT)){
			throw new ParseException(String.format("The option '%s' is required.", UPDATE_RULE_OPT));
		}
		String updateRuleName = cmdLine.getOptionValue(UPDATE_RULE_OPT);
		try{
			updateRule = Enum.valueOf(MMPUpdateRuleType.class, updateRuleName);
		}catch(Exception ex){
			throw new ParseException(String.format("The update rule value '%s' is not valid.", updateRuleName));
		}
		
		MMPLearner learner = cmdLine.hasOption(CommonOptions.SEED_OPT) ? 
				new MMPLearner(loss, updateRule, ((Number)cmdLine.getParsedOptionValue(CommonOptions.SEED_OPT)).longValue()) : 
				new MMPLearner(loss, updateRule);
		
		if(cmdLine.hasOption(NOMINAL_TO_BINARY_OPT)){
			learner.setConvertNominalToBinary(true);
		}else{
			learner.setConvertNominalToBinary(false);
		}

        return learner;
	}
	   
}
