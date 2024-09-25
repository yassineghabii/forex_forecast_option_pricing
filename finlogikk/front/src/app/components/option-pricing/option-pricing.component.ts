import { Component } from '@angular/core';
import {OptionPricingService} from "../../services/option-pricing/option-pricing.service";

@Component({
  selector: 'app-option-pricing',
  templateUrl: './option-pricing.component.html',
  styleUrls: ['./option-pricing.component.css']
})
export class OptionPricingComponent {

  results: any;
  error: string;

  formData = {
    option_type: 'c',
    ticker: 'USDJPY=X',
    k: 110.0,
    t: 0.5,
    rf: -0.1,
    rd: 0.05
  };

  tickers = [
    'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X'
  ];

  constructor(private optionPricingService: OptionPricingService) { }

  onSubmit(): void {
    this.optionPricingService.calculateOption(this.formData).subscribe(
      data => {
        this.results = data;
        this.error = null;
        console.log(this.results);
      },
      error => {
        this.error = 'Erreur lors du calcul: ' + error.message;
        this.results = null;
      }
    );
  }

}
