package WEKA;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Cositas {

    public static void main(String[] args) throws Exception {
        DataSource source = new ConverterUtils.DataSource("/home/adeiarias/Escritorio/3.MAILA/2.LAUHILABETEA/WEKA/datasets/vote.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);


        //RANDOMIZE
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        randomize.setRandomSeed(1);
        Instances random = Filter.useFilter(data,randomize);

        //SPLIT
        RemovePercentage removePercentage = new RemovePercentage();
        removePercentage.setInputFormat(random);
        removePercentage.setInvertSelection(true);
        removePercentage.setPercentage(70);
        Instances train = Filter.useFilter(random,removePercentage);
        removePercentage.setInputFormat(random);
        removePercentage.setInvertSelection(false);
        removePercentage.setPercentage(70);
        Instances test = Filter.useFilter(random,removePercentage);

        //MODEL SORTU
        NaiveBayes naive = new NaiveBayes();
        train.setClassIndex(train.numAttributes()-1);
        naive.buildClassifier(train);

        //EVALUATION
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naive,test);
        System.out.println(eval.pctCorrect());
    }
}
