import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-services',
  templateUrl: './services.component.html',
  styleUrls: ['./services.component.css'],
})
export class ServicesComponent implements OnInit {
  services = [
    {
      icon: 'fas fa-calculator',
      heading: 'Comprehensive Option Pricing',
      description: 'Leverage advanced option pricing models, including the Garman-Kohlhagen model, to calculate accurate option values. Our tool provides insights into the Greeks (Delta, Gamma, Theta, and more) to help traders manage risk and optimize their strategies in the options market.'
    },
    {
      icon: 'fas fa-chart-line',
      heading: 'Accurate Exchange Rate Forecast',
      description: 'Predict currency movements with our highly precise model, achieving an R² score of 99%. In addition to accurate forecasts, our tool detects various trading patterns, empowering traders to identify opportunities and make informed decisions with confidence.'
    },
    {
      icon: 'fas fa-globe',
      heading: 'Forex Dashboard with Advanced Insights',
      description: 'Stay ahead in the Forex market with our Forex Dashboard. Track currency pair prices, review technical analysis, and access comprehensive indicators, all designed to give you the upper hand in making well-informed trading choices.'
    },
    {
      icon: 'fas fa-comments',
      heading: 'Sentiment Analysis for Financial News',
      description: 'Understand market sentiment with our real-time news sentiment analysis tool. It filters and processes financial news, providing you with a clear overview of market mood and helping you anticipate market movements based on the latest news trends.'
    },
    {
      icon: 'fas fa-users',
      heading: 'Interactive Financial Forum',
      description: 'Join our vibrant community of traders and finance enthusiasts in our interactive forum. Share insights, discuss strategies, and learn from others about the latest trends in trading, investment, and market analysis, all in one collaborative space.'
    }
  ];
  constructor() {}

  ngOnInit(): void {}
}
