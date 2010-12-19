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
import mulan.console.CommonOptions;
import mulan.core.ArgumentNullException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * Command line interface builder for {@link BPMLL} learner.
 * 
 * @see LearnerBuilder
 * @author Jozef Vilcek
 */
public class BPMLLBuilder implements LearnerBuilder {

	@SuppressWarnings("static-access")
	@Override
	public Options getOptions() {
		Options opt = new Options();
		
		opt.addOption(OptionBuilder.withLongOpt("hidden-layers").hasArgs().withValueSeparator(',')
				.withDescription("Sets the topology of hidden layers for the neural network. Enter the coma separated list of integer values, whenre number of values will define number of hidden layers and each value defines number of neurons in the particular layer. If empty value is passed, no hidden layers will be created.")
				.create("hl"));
		opt.addOption(OptionBuilder.withLongOpt("learning-rate").hasArg().withType(Number.class)
				.withDescription("Sets the learning rate. The value must be greater than 0 and no more than 1.")
				.create("lr"));
		opt.addOption(OptionBuilder.withLongOpt("weights-decay").hasArg().withType(Number.class)
				.withDescription("Sets the regularization cost term for weights decay. The value must be greater than 0 and no more than 1.")
				.create("wd"));
		opt.addOption(OptionBuilder.withLongOpt("training-epochs").hasArg().withType(Number.class)
				.withDescription("Sets the number of training epochs. The value must be integer greater than 0.")
				.create("te"));
		opt.addOption(OptionBuilder.withLongOpt("normalize-inputs")
				.withDescription("If defined, all input numeric attributes of the training data set (except label attributes) will be normalized prior to learning to the range <-1,1>.")
				.create("n"));
		opt.addOption(CommonOptions.getSeedOption());
		
		return opt;
	}

	@Override
	public Class<? extends MultiLabelLearner> getSupportedType(){
		return BPMLL.class;
	}

	@Override
	public MultiLabelLearner build(CommandLine cmdLine) throws ParseException {
		if(cmdLine == null){
			throw new ArgumentNullException("cmdLine");
		}
		
		BPMLL learner = cmdLine.hasOption(CommonOptions.SEED_OPT) ? 
				new BPMLL(((Number)cmdLine.getParsedOptionValue(CommonOptions.SEED_OPT)).longValue()) : 
				new BPMLL();
		
		if(cmdLine.hasOption("hl")){
			String[] values = cmdLine.getOptionValues("hl");
			int[] layers = new int[values.length];
			try{
			for(int i = 0; i < values.length; i++){
				layers[i] = Integer.parseInt(values[i]);
			}
			}catch(NumberFormatException ex){
				throw new ParseException(String.format("Values for hiddnen layers are invalid: '%s'", Arrays.toString(values)));
			}
			learner.setHiddenLayers(layers);
		}
		if(cmdLine.hasOption("lr")){
			learner.setLearningRate(((Number)cmdLine.getParsedOptionValue("wd")).doubleValue());
		}	
		if(cmdLine.hasOption("wd")){
			learner.setWeightsDecayRegularization(((Number)cmdLine.getParsedOptionValue("wd")).doubleValue());
		}
		if(cmdLine.hasOption("te")){
			learner.setTrainingEpochs(((Number)cmdLine.getParsedOptionValue("te")).intValue());
		}
		if(cmdLine.hasOption("n")){
			learner.setNormalizeAttributes(true);
		}else{
			learner.setNormalizeAttributes(false);
		}

        return learner;
	}
	   
}
