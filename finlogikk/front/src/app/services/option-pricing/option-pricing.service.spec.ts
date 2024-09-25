import { TestBed } from '@angular/core/testing';

import { OptionPricingService } from './option-pricing.service';

describe('OptionPricingService', () => {
  let service: OptionPricingService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(OptionPricingService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
