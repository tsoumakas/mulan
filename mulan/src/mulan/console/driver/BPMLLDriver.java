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

package mulan.console.driver;

import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.neural.BPMLL;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class BPMLLDriver extends LearnerDriverBase {

	@SuppressWarnings("static-access")
	@Override
	public Options getOptions() {
		Options opt = super.getOptions();
		
		opt.addOption(OptionBuilder.withLongOpt("hidden-layers").hasArgs().withValueSeparator(',')
				.withDescription("Sets the topology of hidden layers for the neural network. Enter the coma separated list of integer values, whenre number of values will define number of hidden layers and each value defines number of neurons in the particular layer. If empty value is passed, no hidden layers will be created.")
				.create("hl"));
		opt.addOption(OptionBuilder.withLongOpt("learning-rate").hasArg().withArgName("value").withType(Number.class)
				.withDescription("Sets the learning rate. The value must be greater than 0 and no more than 1.")
				.create("lr"));
		opt.addOption(OptionBuilder.withLongOpt("weights-decay").hasArg().withArgName("value").withType(Number.class)
				.withDescription("Sets the regularization cost term for weights decay. The value must be greater than 0 and no more than 1.")
				.create("wd"));
		opt.addOption(OptionBuilder.withLongOpt("training-epochs").hasArg().withArgName("value").withType(Number.class)
				.withDescription("Sets the number of training epochs. The value must be integer greater than 0.").create("te"));
		opt.addOption(OptionBuilder.withLongOpt("normalize-inputs")
				.withDescription("If defined, all input numeric attributes of the training data set (except label attributes) will be normalized prior to learning to the range <-1,1>.")
				.create("n"));
		
		return opt;
	}

	@Override
	public List<Class<? extends MultiLabelLearner>> getSupportedTypes() {
		List<Class<? extends MultiLabelLearner>> list = new ArrayList<Class<? extends MultiLabelLearner>>();
		list.add(BPMLL.class);
		
		return list;
	}

	@Override
	protected MultiLabelLearner createLearner(CommandLine cmdLine) {
		
	        // TODO: add param parsing and setup of the learner 
	        return new BPMLL();
	}
	   
}
