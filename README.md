# **Exploring NYC Rental Property Data**

# Introduction and Data Retrieval

Throughout NYC, a vast population of residents rely on rental properties for housing, which can vary greatly based on location, the number of bedrooms and bathrooms, as well as price. With this data project, I aimed to explore NYC rental property data to derive insights regarding apartment data, fostering data-driven decision making on where a user should best look for a rental property based on their specifications. With this project, I attempt to:


● Analyze a key factor, like location

● Identify patterns and trends in NYC rental property data within isochronic
shapes surrounding specified locations

● Build multiple models to help gain insights on determining apartment recommendations (work, school, gym, etc.)




---


The notebook is organized in the following order:

1.   Introduction and Data Retrieval
2.   Setup and Installation
3.   Data Pre-Processing
4.   General Dataset Exploration and Analysis
5.   TravelTime API Function Implementation
6.   Isochrone and Intersection Mapping
7.   Isochrone Data Analysis
8.   Final Thoughts and Conclusion

To acquire this data on NYC rental properties, I used a data set, scraped from Zillow, a cloud platform for MLS data. The data was collected using web scraping techniques via a custom scraper. This data was scraped on April 29, 2024 and was validated via URLs directly found within the scraped csv file.




# Data Pre-Processing

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/DPP1.png">

Looking through the columns, I first looked at the columns: `address`, `area`, `units/{i}/beds`, `baths`, `units/{i}/price`, `latLong/latitude`, and `latLong/longitude`. For clarification, if a listing has "0" beds, it simply means the apartment is a studio and has no separate rooms for the bed.

 I first decided to check for the NA count for columns, to see whether or not they would be viable to use in my analysis later on. I will not be looking at `units/{i}/beds` or `units/{i}/price` as apartments will vary in the number of bedrooms.

 <img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/DPP2.png">

 Based on my tentative columns for my final dataframe, I see that out of 3486 rows, 2836 do not have data for `area`. Thus, I will not use it. Now that I know the relevant columns, I can move onto further pre-processing. As for missing values in `baths`, I know that apartments MUST have a bathroom in the vicinity, and thus will default the value as 1. After cleaning up the data table, the dataframe has a size of (5556, 6).

Now that I have reformatted the data so that each variable is now separate and organized in its proper column, I did further pre-processing by removing NaN values, as well as implementing feature engineering by adding columns `neighborhood` (derived from latitude and longitude) and `total rooms` (derived from adding the number of bedrooms and bathrooms with a "base room" like a living room) to which I then received a size of (5507,6).

After a complete cleaning of the dataframe, I received the following shape and dataframe:

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/DPP3.png">

After pre-processing, I have come up with the dataframe I plan to use in my analysis and model building. The DataFrame has 5398 rows and 9 columns. As a side note, I am omitting from normalizing the columns just yet, as interpreting my analysis will be more tangible with the actual prices. I will normalize the data for regression and clustering later when I analyze isochrone intersection apartment data.



# General Dataset Exploration and Analysis

Before going into the exploring and analyzing of apartments within isochrone intersections, I plan on comprehensively exploring the whole dataset. For each of the numerical categories, I provided descriptive statistics and plotted a histogram depicting the distribution of values in each column.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA1.png">

DescribeResult(nobs=5398, minmax=(0.0, 6.0), mean=2.037791774731382, variance=1.3130054619149154, skewness=0.3687933300586163, kurtosis=-0.0390447557102247)


Based on the descriptive statistics and histogram visualizing the distribution of 'beds', the data appears to be normal at an average of 2.04 beds for an apartment. While the data is relatively normal, there is a slight skew to the right as the number of bedrooms in an apartment increases, with the mean being greater than the median. To check again, I can use stats.describe to better understand the distribution of data.

Using stats.describe, I get low skewness values, suggesting that the column `beds` has a useable normal distribution.

*For clarification, I should also interpret '0 beds' to mean 'studio'.*

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA2.png">

Upon looking at the descriptive statistics and histogram depicting distribution of 'baths' data, I can see that the data is heavily skewed to the right. Not only is the mean greater than the median (1.46 > 1), but as the number of bathrooms increase, the count drastically drops. This suggests that the majority of apartments only have one bathroom, less have two, even less have three, and so on. The histogram also highlights the presence of "half bathrooms", bathrooms that do not house full amenities like a shower.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA3.png">
DescribeResult(nobs=5398, minmax=(1600.0, 32000.0), mean=6752.831789551686, variance=24526043.251670357, skewness=2.240537968962258, kurtosis=5.692106147734787)

Like the `baths` column, `price` is also heavily skewed to the right. The mean is greater than the median price, and the histogram's shape suggests the same distribution. I find an existing pattern that as the price goes up, the number of existing apartments decreases.

Running stats.describe on the `price` column, I also find that there is very high variance, a high positive skewness, and high kurtosis values, confirming that the distribution is heavily right-tailed and not normal.

I can interpret this distribution to mean that most apartments hover around a median of $5000, with the count of apartments decreasing as they increase in price.
I also looked at `neighborhood` data, grouping by the column. I aggregated counts and observed boxplots of price for each of these neighborhoods to better understand the data.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA4.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA5.png">

Based on the bar chart, the top 5 most common neighborhoods in the NYC Rental Property dataset are: Carnegie Hall, Clinton, East Village, Murray Hill, and Upper East Side.

Observing the boxplots that visualize the distribution and summary statistics of each of these neighborhoods, I can also make several inferences. For one, Carnegie Hall has a wide range of prices for its apartments, with a large amount of outliers at the top, indicating luxury housing. This wide range can indicate that there is wide variability in terms of housing, potentially based on if it is standard or luxury housing. In another instance, I see that Clinton has a very uniform interquartile range, suggesting that the rent is fairly uniform across most apartments, in spite of some outliers. In a neighborhood like the Upper East side, the narrow bottom 50% of data, contrasted with the wide variability and outliers in the top 50% of data, suggests that standard, lower-priced apartments are fairly narrowly distributed in price, while higher-priced luxury apartments can vary signficiantly more.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/GDA6.png"> 

Corr between # of beds and price:  PearsonRResult(statistic=-0.02350657776258395, pvalue=0.08418699985720486)
Corr between # of baths and price:  PearsonRResult(statistic=0.5905038428225113, pvalue=0.0)
Corr between # of total rooms and price:  PearsonRResult(statistic=0.5877040482350792, pvalue=0.0)

Based on the correlation ran between the three numerical variables `beds`, `baths`, `total rooms`, and `price`, I notice that all three instances of Pearson correlations produce a statistic value of 0.53, 0.59, and 0.59. These values indicate a moderately strong positive direct relationship for all three pairs of variables. The p-value being listed as 0.0 for each of these correlations corroborates this relationship, as the value being less than 0.05 means I can reject the null hypothesis that the correlation coefficient between these variables is 0. I expected these values, as it makes sense for the price of an apartment to go up with the increase of area and utility in the form of bedrooms and bathrooms.

I can run significance tests to explore further questions related to my dataset. For instance, I would first like to run a 2-sample t-test to identify if there is a statistically significant difference between two distributions of bed counts between the neighborhoods 'East Village' and 'Murray Hill'.

The results are: TtestResult(statistic=5.589021145989148, pvalue=3.236793151667449e-08, df=723.0)

I can interpret the results of this two-sample t-test by looking at the p-value. Since the p-value is 3.24*10^-8, which is less than an alpha value of 0.05, I reject the null hypothesis that there is no statistically significant difference in means between the two neighborhoods East Village and Murray Hill.

On a side note, while I did want to conduct an ANOVA to compare prices between certain neighborhoods, since price is not normally distributed, I was not able to use the test. That said, non-parametric tests were not in the course notes, so I thus operated under the assumption that non-parametric tests like Kruskal-Wallis, Mann-Whitney U were not allowed.


# Isochrone and Intersection Mapping

Using the added TravelTime API functions, I can create an ischrone map, a map that portrays the area accessible to a certain location within a certain timeframe, for a number of listed locations. In this section, I will demo a user input in which:
*   The user would like to access their locations within 30 minutes by public_transport.
*   Their listed locations are:
  *   Morgan Stanley NY Office (1585 Broadway, New York, NY 10036)
  *   Washington Square Park
  *   Columbus Circle

*As a side note, the parentheticals for locations refer to how the user inputs the information. Inputting the full address or a common name can better help identify addresses.*

The result will display geographic visualizations of the isochrones for each location. Using these isochrones, I can later look at their intersections, and the apartment data within this intersection.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IIM1.png">

Now that I have three locations plotted to an individual color, I can now plot the intersection between these isochrones.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IIM2.png"> 

From here, I can now plot apartments from my dataframe into the isochrone intersection, as well as produce a dataframe of said apartments for later analysis.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IIM3.png">




# Isochrone Data Analysis

I can run some analysis on the dataset of apartments within this intersection of isochrones, as well as produce visualizations.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA1.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA2.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA3.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA4.png">

Looking at the boxplots of price distributions by neighborhood, I can first identify that Clinton has a fairly narrow range of prices compared to the other four neighborhoods. East Village and Murray Hill share similarities both in boxplot shape, suggesting similar distribution, and the presence of outliers. And like my observation in exploring the general dataframe, I also find that the Upper East side has a very narrow section for its bottom half of data, with a very high range from the maximum (excluding outlier) to the median, suggesting a wide range of luxury housing in the area.

The boxplot of rental property grades within the 5 most common neighborhoods provides good insight into the quality of the applicable apartments. Clinton, being the most common neighborhood, has the highest grade of apartments. Although Clinton has the highest grades, it is noted that the average grade is found just below that of East Village. Despite the top 5 neighborhoods all having relatively similar median apartment grades, it would be recommended that the buyer looks for apartments in the neighborhoods of Clinton, East Village, and Murry Hill as those neighborhoods have the highest average grade as well as the absolute highest grades.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA5.png">

Looking at the correlation between beds and price, a range of values of `0.59` to `0.64` indicates a moderately positive relationship between variables `beds` and `price`. In addition, p-values being listed at `0.00` show that the correlation is statistically significant, which again, I have come to expect.

The following are regression analyses to determine the relationships between different variables, such as price and total rooms (the first regression), price and neighborhood (the second regression), and grade and neighborhood (the third regression).

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA6.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA7.png">

Analysis of the Regression Model:

The training error starts relatively high and decreases significantly as the training set size increases up to around 1500 samples. This decrease indicates that the model benefits from more training data, improving its ability to predict the training set. Post-1500 samples, the training error experiences some fluctuation but tends to decrease, suggesting some variability in how additional data impacts model training.

The validation error starts lower than the training error, which is unusual as typically, validation error will start higher. This could indicate an issue with how the validation set is composed and or how the model is evaluated. The validation error remains relatively flat and even slightly increases as more data is added. It shows less variability than the training error, indicating that the model's generalization to unseen data isn't substantially improving with more data.

The gap between the training and validation errors is relatively narrow throughout, which typically suggests low variance in the model predictions. A narrow gap usually indicates good generalization but can also suggest underfitting if both errors are high.

In my case, a mean squared error of 0.015-0.019 is moderate. Thus, the regression model is demonstrating some signs of underfitting for the data, as it doesn’t fully learn from the data. This is also reflected in the R^2 score. As less than 50% of the variance in price is accounted for within total room count.

<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA8.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA9.png">
<img src="https://github.com/CallEmUp/TimeTravelRental/blob/main/images/IDA10.png">

# Final Thoughts and Conclusion

**Challenges and Notes for Future**

One of the main challenges I came across was the distribution of my numerical data, specifically how columns like `price` were skewed to the right. Since the distribution to my y-value is not normal, I did not have the necessary parameters to use an ANOVA test to compare a categorical variable like `neighborhood` with `price`. Based on my notebook not containing a non-parametric equivalent to this test (Kruskal-Wallis), in the future, I would resort to using other, non-parametric tests, as rental property price data shows a distribution of being heavily skewed.

I also encountered issues with the creation of linear regression models. As I were creating them, I ran into issues of over and underfitting which suggested that my model did not have enough information to make a sufficient model. These issues posed a challenge when I interpreted and analyzed my linear regression model. In the future, I aim to change the testing size and increase data size, while also adjusting the size based on evaluation, to get a better fitting model. In the future, I also aim to train on more features, as the lack of features is a probable cause for the underfitting. This fix along with some changes to the testing and data size will allow us to get a better fitting model

**Recommendation**

One of the main insights that I gathered from this project was that although distributions of price can be visually similar between neighborhoods, that it was not indicative of similarly structured apartments. To clarify, while the distribution of price via boxplots in neighborhoods Murray Hill and East Village were visually congruous, a two sample t-test concerning the number of beds between these two neighborhoods produced a p-value that rejected my null hypothesis that the mean bed numbers for each neighborhood were not statistically significantly different. Thus, I recommend that viewing apartments by price will not mean that the distribution of beds will also be similar.

The first thing I looked at in terms of recommendations for where the user should look for an apartment is the most common neighborhoods. When searching for apartments, quantity matters. There are a wide variety of other requirements people may have for their place of living so it is good to have the option to look at as many apartments as possible. Within the most common neighborhoods, the next most important thing that many consumers have is how good is the space for what they are paying. This was analyzed through the boxplots of property grade in the most common neighborhoods. The 3 beds were determined to have the best grade, suggesting that 3 bedroom apartments are the best value for money. Not only are they the best value for money, but also within the neighborhoods of Clinton and East Village. These neighborhoods have a good market too, as the median price within the neighborhood is roughly 5000 USD for Clinton and 6000 USD for East Village, compared to the overall median price of 5000 USD across all apartment types. Based on these insights, I would recommend that the user gather 2 other friends and look for a 3 bedroom apartment within the neighborhoods of Clinton and East Village.
